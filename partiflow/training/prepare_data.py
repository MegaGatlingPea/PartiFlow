"""Explicit PepInter extraction, split indexing and atom37 graph preparation."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import tempfile

SPLITS = {"train": "train", "valid": "val", "test": "test"}
EXPECTED_COUNTS = {"train": 166019, "val": 3750, "test": 1250}
FIELDS = ("pdbid", "split", "tag", "structure_path", "processed_path")


def read_prepared_index(root: Path, *, require_files: bool = True) -> tuple[list[dict], dict]:
    """Read the dataset split and its referenced PDB/graph files without loading graphs."""
    root = Path(root).resolve()
    index = root / "indexes" / "split.csv"
    with index.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or set(reader.fieldnames) != set(FIELDS):
            raise ValueError("PepInter split.csv must contain the standard split fields")
        rows = list(reader)
    if not rows:
        raise ValueError("PepInter split.csv is empty")
    counts = {split: 0 for split in SPLITS.values()}
    seen = set()
    for row in rows:
        identifier, split = row["pdbid"], row["split"]
        if not identifier or identifier in seen or split not in counts or not row["tag"]:
            raise ValueError("PepInter split.csv has an invalid or repeated sample")
        seen.add(identifier)
        counts[split] += 1
        directory = "valid" if split == "val" else split
        expected = {
            "structure_path": root / "raw" / directory / (identifier + ".pdb"),
            "processed_path": root / "processed" / split / (identifier + ".pt"),
        }
        for field, target in expected.items():
            value = row[field]
            if not value or Path(value).is_absolute() or "\\" in value:
                raise ValueError(f"PepInter {field} must be a dataset-relative path")
            actual = (index.parent / value).resolve()
            if not actual.is_relative_to(root) or actual != target.resolve():
                raise ValueError(f"PepInter {field} does not match sample {identifier}")
            if require_files and (not actual.is_file() or actual.stat().st_size == 0):
                raise FileNotFoundError(f"PepInter sample file is missing or empty: {actual}")
    return rows, counts


def _prepared_result(root: Path, counts: dict) -> dict:
    return {
        "counts": counts,
        "split_csv": str(Path(root) / "indexes" / "split.csv"),
        "prepared": True,
    }


def is_prepared_archive(archive: Path) -> bool:
    """Identify the layout from early members without scanning a full raw archive."""
    with tarfile.open(archive, "r|*") as handle:
        for member in handle:
            parts = PurePosixPath(member.name).parts
            if len(parts) < 2 or parts[0] != "pepinter":
                continue
            if parts[1] in SPLITS:
                return False
            if parts[1] in {"raw", "processed", "indexes"}:
                return True
    return False


def _copy_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + destination.name, suffix=".tmp", dir=destination.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as writer, Path(source).open("rb") as reader:
            shutil.copyfileobj(reader, writer)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def publish_prepared_dataset(source: Path, destination: Path, *, force: bool = False) -> dict:
    """Publish validated prepared files, keeping an existing split table until last."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    rows, counts = read_prepared_index(source)
    old_index = destination / "indexes" / "split.csv"
    if old_index.exists() and not force:
        try:
            previous, _ = read_prepared_index(destination, require_files=False)
        except (OSError, ValueError) as exc:
            raise ValueError(
                "The existing PepInter index cannot be reused; select a new --data-dir or use --force"
            ) from exc
        if {row["pdbid"]: row for row in previous} != {row["pdbid"]: row for row in rows}:
            raise ValueError(
                "The existing PepInter index differs from this dataset; select a new --data-dir or use --force"
            )
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        source.replace(destination)
        return _prepared_result(destination, counts)
    if not destination.is_dir():
        raise ValueError("The PepInter destination must be a directory")
    files = [path for path in source.rglob("*") if path.is_file()]
    files.sort(key=lambda path: (path.relative_to(source).as_posix() == "indexes/split.csv",
                                 path.relative_to(source).as_posix()))
    for path in files:
        target = destination / path.relative_to(source)
        if not target.resolve().is_relative_to(destination):
            raise ValueError("An existing PepInter path points outside the dataset directory")
    written = []
    with tempfile.TemporaryDirectory(prefix=".pepinter-backup-", dir=destination.parent) as temporary:
        backups = Path(temporary)
        try:
            for path in files:
                target = destination / path.relative_to(source)
                if target.is_file() and target.stat().st_size > 0 and not force:
                    continue
                backup = None
                if target.exists():
                    backup = backups / str(len(written))
                    shutil.copy2(target, backup)
                written.append((target, backup))
                _copy_atomic(path, target)
        except BaseException:
            for target, backup in reversed(written):
                if backup is None:
                    target.unlink(missing_ok=True)
                else:
                    _copy_atomic(backup, target)
            raise
    return _prepared_result(destination, counts)


def install_prepared_archive(
    archive: Path, root: Path, *, force: bool = False, archive_name: str | None = None
) -> dict:
    """Extract and publish a prepared dataset while leaving its source archive intact."""
    root = Path(root).resolve()
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pepinter-prepare-", dir=root.parent) as temporary:
        stage = Path(temporary) / "pepinter"
        safe_extract_archive(Path(archive), stage)
        if archive_name:
            shutil.copy2(archive, stage / archive_name)
        return publish_prepared_dataset(stage, root, force=force)


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
        if is_prepared_archive(Path(archive)):
            return install_prepared_archive(Path(archive), root)
        safe_extract_archive(Path(archive), root / "raw")
    elif (root / "indexes" / "split.csv").is_file():
        _, counts = read_prepared_index(root, require_files=False)
        if counts != EXPECTED_COUNTS:
            read_prepared_index(root)
            return _prepared_result(root, counts)
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
