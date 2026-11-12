"""Explicit PepInter extraction, split indexing and atom37 graph preparation."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile

SPLITS = {"train": "train", "valid": "val", "test": "test"}
EXPECTED_COUNTS = {"train": 166019, "val": 3750, "test": 1250}
FIELDS = ("pdbid", "split", "tag", "structure_path", "processed_path")


def safe_extract_archive(archive: Path, raw_dir: Path) -> int:
    """Extract regular files under the archive's single pepinter directory."""
    archive, raw_dir = Path(archive).resolve(), Path(raw_dir).resolve()
    raw_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(archive, "r|xz") as handle:
        for member in handle:
            parts = PurePosixPath(member.name).parts
            if (
                not parts
                or parts[0] != "pepinter"
                or ".." in parts
                or "\\" in member.name
            ):
                raise ValueError(f"Unsafe archive member: {member.name}")
            if len(parts) == 1:
                if not member.isdir():
                    raise ValueError(f"Invalid archive root: {member.name}")
                continue
            destination = raw_dir.joinpath(*parts[1:])
            if not destination.resolve().is_relative_to(raw_dir):
                raise ValueError(f"Archive member escapes raw directory: {member.name}")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile() or destination.is_symlink():
                raise ValueError(
                    f"Archive member must be a regular file: {member.name}"
                )
            destination.parent.mkdir(parents=True, exist_ok=True)
            source = handle.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read archive member: {member.name}")
            temporary = destination.with_name(destination.name + ".extracting")
            try:
                with source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if temporary.stat().st_size != member.size:
                    raise ValueError(f"Incomplete archive member: {member.name}")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            count += 1
    return count


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def build_indexes(root: Path, *, verify_counts: bool = True) -> tuple[Path, Path, dict]:
    """Keep source splits and select fixed sorted 64 train / 16 val rows."""
    root = Path(root).resolve()
    index_dir = root / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)
    rows, counts = [], {}
    for directory, split in SPLITS.items():
        structures = sorted(
            (root / "raw" / directory).glob("*.pdb"), key=lambda item: item.stem
        )
        counts[split] = len(structures)
        for structure in structures:
            rows.append(
                {
                    "pdbid": structure.stem,
                    "split": split,
                    "tag": "pepinter",
                    "structure_path": os.path.relpath(structure, index_dir),
                    "processed_path": os.path.relpath(
                        root / "processed" / split / (structure.stem + ".pt"), index_dir
                    ),
                }
            )
    if verify_counts and counts != EXPECTED_COUNTS:
        raise ValueError(
            f"PepInter split counts differ: expected {EXPECTED_COUNTS}, found {counts}"
        )
    identifiers = [row["pdbid"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("PepInter identifiers occur in multiple source splits")
    full_path = index_dir / "split.csv"
    subset_path = index_dir / "subset.csv"
    _write_rows(full_path, rows)
    subset = []
    for split, size in (("train", 64), ("val", 16)):
        selected = [row for row in rows if row["split"] == split][:size]
        if verify_counts and len(selected) != size:
            raise ValueError(
                f"Insufficient rows for the fixed {split} subset"
            )
        subset.extend(selected)
    _write_rows(subset_path, subset)
    return full_path, subset_path, counts


def prepare_graphs(root: Path, index_csv: Path) -> None:
    # Import the graph stack only after extraction and indexing have succeeded.
    from partiflow.datasets.pdb_data import PepLightningDataModule

    module = PepLightningDataModule(
        split_csv=str(index_csv),
        processed_dir=str(Path(root) / "processed"),
        structure_dir=str(Path(root) / "raw"),
        pep_chain="L",
        num_workers=0,
    )
    module.prepare_data(
        skip=False, failure_csv=str(Path(root) / "indexes" / "preparation_failures.csv")
    )


def prepare(
    root: Path,
    archive: Path | None = None,
    *,
    subset: bool = True,
    extract: bool = True,
    graphs: bool = True,
) -> dict:
    root = Path(root).resolve()
    if extract:
        if archive is None:
            raise ValueError("An archive path is required for extraction")
        safe_extract_archive(Path(archive), root / "raw")
    full, selected, counts = build_indexes(root)
    if graphs:
        prepare_graphs(root, selected if subset else full)
    return {"counts": counts, "split_csv": str(full), "subset_csv": str(selected)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--subset",
        action="store_true",
        help="Prepare the fixed 64 train / 16 val subset rows",
    )
    parser.add_argument("--skip-extract", action="store_true")
    parser.add_argument(
        "--index-only",
        action="store_true",
        help="Extract and index without importing Torch or Graphein",
    )
    args = parser.parse_args(argv)
    print(
        prepare(
            args.root,
            args.archive,
            subset=args.subset,
            extract=not args.skip_extract,
            graphs=not args.index_only,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
