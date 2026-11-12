"""Explicit model/data acquisition with atomic local and URL imports."""

from __future__ import annotations
import json
import os
import shutil
import stat
import tarfile
import tempfile
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from partiflow.paths import model_root
from partiflow.ui import DownloadDisplay, RESOURCE_USAGE, message, show_inventory, terminal_output

MPNN_URL = "https://raw.githubusercontent.com/dauparas/ProteinMPNN/8907e6671bfbfc92303b5f79c4b5e6ce47cdef57"
AF2_URL = "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"


@dataclass(frozen=True)
class Asset:
    relative: str
    url: str | None = None
    aliases: tuple[str, ...] = ()


TARGETS = {
    name: (Asset(f"ckpts/{name}.ckpt"),)
    for name in (
        "partiflow",
        "partiflow-aux",
        "partiflow-dock",
        "partiflow-ss",
        "autoencoder",
    )
}
TARGETS["af2"] = tuple(
    Asset(f"af2/params_model_{number}{suffix}.npz")
    for number in range(1, 6)
    for suffix in ("", "_ptm", "_multimer_v3")
)
TARGETS["proteinmpnn"] = (
    Asset(
        "proteinmpnn/vanilla/v_48_020.pt",
        f"{MPNN_URL}/vanilla_model_weights/v_48_020.pt",
        (
            "vanilla/v_48_020.pt",
            "community_models/proteinmpnn/checkpoint/vanilla/v_48_020.pt",
        ),
    ),
)
TARGETS["solublempnn"] = (
    Asset(
        "proteinmpnn/soluble/v_48_020.pt",
        "https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_020.pt",
        (
            "soluble/solublempnn_v_48_020.pt",
            "soluble/v_48_020.pt",
            "solublempnn_v_48_020.pt",
            "community_models/proteinmpnn/checkpoint/soluble/v_48_020.pt",
        ),
    ),
)
MODEL_TARGETS = tuple(TARGETS)
TARGETS["pepinter"] = ()


def data_root(root, *, data_dir=None, config=None):
    if data_dir:
        return Path(data_dir).expanduser().resolve()
    if config:
        from partiflow.training import config as training_config

        path = Path(config).expanduser().resolve()
        value = training_config.load_config(path, model_directory=root)
        selected = value.get("data_dir")
        if selected:
            selected = Path(selected).expanduser()
            if not selected.is_absolute():
                base = (
                    training_config.default_work_root()
                    if path.is_relative_to(training_config.config_dir().resolve())
                    else path.parent
                )
                selected = base / selected
            return selected.resolve()
        dataset = value.get("dataset") or {}
        datamodule = dataset.get("datamodule") or value.get("datamodule") or {}
        processed = datamodule.get("processed_dir")
        if processed:
            processed = Path(processed)
            selected = processed.parent.resolve()
            split = datamodule.get("split_csv")
            if processed.name != "processed" or (
                split and Path(split).parent.resolve() != selected / "indexes"
            ):
                raise ValueError(
                    "PepInter resource commands require indexes/ and processed/ in one data root; "
                    "use --data-dir to select a standard-layout destination"
                )
            return selected
    return Path(root) / "train_data" / "pepinter"


def resource_status(target, root, *, data_dir=None, config=None):
    if target not in TARGETS:
        raise ValueError(f"Unknown download target: {target}")
    root = Path(root)
    if target == "pepinter":
        path = data_root(root, data_dir=data_dir, config=config)
        checks = {
            "split_csv": (path / "indexes" / "split.csv").is_file()
            and (path / "indexes" / "split.csv").stat().st_size > 0,
            "processed": (path / "processed").is_dir()
            and any(item.is_file() for item in (path / "processed").rglob("*")),
        }
        return {
            "target": target,
            "status": "present" if all(checks.values()) else "missing",
            "path": str(path),
            "checks": {key: bool(value) for key, value in checks.items()},
        }
    files = [
        {
            "path": str(root / asset.relative),
            "present": (root / asset.relative).is_file()
            and (root / asset.relative).stat().st_size > 0,
        }
        for asset in TARGETS[target]
    ]
    return {
        "target": target,
        "status": "present" if all(item["present"] for item in files) else "missing",
        "files": files,
    }


def _safe_member(name):
    name = str(name).replace("\\", "/")
    path = PurePosixPath(name)
    if (
        not name
        or "\x00" in name
        or path.is_absolute()
        or ".." in path.parts
        or (path.parts and ":" in path.parts[0])
    ):
        raise ValueError(f"Unsafe archive member: {name!r}")
    return path


def extract_archive(source, destination):
    """Extract regular files only, after validating every archive member."""
    source, destination = Path(source), Path(destination)
    if zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            entries = archive.infolist()
            for entry in entries:
                _safe_member(entry.filename)
                mode = entry.external_attr >> 16
                if stat.S_ISLNK(mode) or (
                    stat.S_IFMT(mode) and not (stat.S_ISREG(mode) or stat.S_ISDIR(mode))
                ):
                    raise ValueError(
                        f"Archive links/special files are unsupported: {entry.filename}"
                    )
            for entry in entries:
                path = destination / str(_safe_member(entry.filename))
                if entry.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(entry) as reader, path.open("wb") as writer:
                        shutil.copyfileobj(reader, writer)
    elif tarfile.is_tarfile(source):
        with tarfile.open(source) as archive:
            entries = archive.getmembers()
            for entry in entries:
                _safe_member(entry.name)
                if not (entry.isfile() or entry.isdir()):
                    raise ValueError(
                        f"Archive links/special files are unsupported: {entry.name}"
                    )
            for entry in entries:
                path = destination / str(_safe_member(entry.name))
                if entry.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                else:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with (
                        archive.extractfile(entry) as reader,
                        path.open("wb") as writer,
                    ):
                        shutil.copyfileobj(reader, writer)
    else:
        raise ValueError(f"Unsupported archive: {source}")


def _is_archive(path):
    return (
        str(path)
        .lower()
        .endswith((".tar", ".tar.gz", ".tgz", ".zip", ".tar.bz2", ".tbz2", ".tar.xz"))
    )


def _find_local(root, asset):
    candidates = [root / asset.relative, root / Path(asset.relative).name]
    candidates.extend(root / name for name in asset.aliases)
    if asset.relative.startswith("af2/"):
        candidates.append(
            root / "community_models" / "colabdesign" / "checkpoint" / asset.relative
        )
    for path in candidates:
        if path.is_file() and path.stat().st_size:
            return path
    matches = [
        path
        for path in root.rglob(Path(asset.relative).name)
        if path.is_file() and path.stat().st_size
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous local asset: {asset.relative}")
    raise FileNotFoundError(f"Local source has no {asset.relative}")


def _fetch(url, destination):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"https", "http", "file"}:
        raise ValueError("URL must use https, http, or file")
    with (
        urllib.request.urlopen(url, timeout=120) as reader,
        Path(destination).open("wb") as writer,
    ):
        expected = reader.headers.get("Content-Length")
        if expected is not None:
            expected = int(expected)
            if expected < 0:
                raise ValueError(f"Invalid content length from {url}")
        shutil.copyfileobj(reader, writer)
    actual = Path(destination).stat().st_size
    if not actual:
        raise ValueError(f"Empty download from {url}")
    if expected is not None and actual != expected:
        raise ValueError(
            f"Incomplete download from {url}: expected {expected} bytes, received {actual}"
        )


def _publish_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix="." + destination.name, suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "wb") as writer, Path(source).open("rb") as reader:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, destination)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _import_pepinter(source, destination, *, force):
    roots = [source, source / "pepinter", source / "train_data" / "pepinter"]
    roots.extend(path.parent for path in source.rglob("indexes") if path.is_dir())
    candidate = next(
        (
            path
            for path in roots
            if (path / "indexes" / "split.csv").is_file()
            and (path / "processed").is_dir()
        ),
        None,
    )
    if candidate is None:
        raise ValueError("PepInter source needs indexes/split.csv and processed/")
    files = [
        path
        for folder in ("indexes", "processed")
        for path in (candidate / folder).rglob("*")
        if path.is_file()
    ]
    if not any(path.is_relative_to(candidate / "processed") for path in files):
        raise ValueError("PepInter processed/ is empty")
    for path in files:
        final = destination / path.relative_to(candidate)
        if force or not final.is_file():
            _publish_file(path, final)
    return destination


def acquire(
    target, root, *, from_local=None, url=None, force=False, data_dir=None, config=None
):
    if target not in TARGETS:
        raise ValueError(f"Unknown download target: {target}")
    root = Path(root).expanduser().resolve()
    existing = resource_status(target, root, data_dir=data_dir, config=config)
    if existing["status"] == "present" and not force:
        return {**existing, "status": "skipped"}
    if (
        not from_local
        and not url
        and target not in {"af2", "proteinmpnn", "solublempnn"}
    ):
        return {
            "target": target,
            "status": "unconfigured",
            "error": "No public URL is configured; use --from-local or --url",
        }
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".partiflow-import-", dir=root
    ) as temporary:
        stage = Path(temporary)
        source = Path(from_local).expanduser().resolve() if from_local else None
        if source and not source.exists():
            raise FileNotFoundError(source)
        source_url = url or (AF2_URL if target == "af2" and source is None else None)
        if source_url:
            source_name = (
                Path(urllib.parse.unquote(urllib.parse.urlsplit(source_url).path)).name
                or "download.bin"
            )
            source = stage / (
                "download.tar" if target == "af2" else "download-" + source_name
            )
            _fetch(source_url, source)
        if (
            source
            and source.is_file()
            and (_is_archive(source) or target in {"af2", "pepinter"})
        ):
            extracted = stage / "extracted"
            extract_archive(source, extracted)
            source = extracted
        if target == "pepinter":
            if source is None or not source.is_dir():
                raise ValueError("PepInter import requires a directory or archive")
            destination = _import_pepinter(
                source, data_root(root, data_dir=data_dir, config=config), force=force
            )
            return {
                **resource_status(target, root, data_dir=destination),
                "status": "imported",
            }
        pending = []
        for index, asset in enumerate(TARGETS[target]):
            destination = root / asset.relative
            if destination.is_file() and destination.stat().st_size > 0 and not force:
                continue
            if source is not None:
                if source.is_file():
                    if len(TARGETS[target]) != 1:
                        raise ValueError(
                            f"{target} requires all configured assets in a directory/archive"
                        )
                    selected = source
                else:
                    selected = _find_local(source, asset)
            else:
                if not asset.url:
                    raise ValueError(f"No configured URL for {asset.relative}")
                selected = stage / f"asset{index}"
                _fetch(asset.url, selected)
            if not selected.stat().st_size:
                raise ValueError(f"Empty source asset: {selected}")
            pending.append((selected, destination))
        for selected, destination in pending:
            _publish_file(selected, destination)
    return {
        **resource_status(target, root),
        "status": "imported" if from_local else "downloaded",
    }


def _selection_options(args, root):
    """Read local paths and source availability without opening model files."""
    options = []
    supplied = bool(getattr(args, "from_local", None) or getattr(args, "url", None))
    for name in TARGETS:
        record = resource_status(name, root, data_dir=getattr(args, "data_dir", None),
                                 config=getattr(args, "config", None))
        configured = name in {"af2", "proteinmpnn", "solublempnn"}
        skipped = record["status"] == "present" and not getattr(args, "force", False)
        source = ("Import source: " + Path(args.from_local).name if getattr(args, "from_local", None) else
                  "Use explicit URL" if getattr(args, "url", None) else
                  "Configured download URL" if configured else
                  "Supply --from-local or --url to acquire this target")
        detail = RESOURCE_USAGE[name] + "\nStatus: " + record["status"] + "\n" + source
        if skipped:
            detail += "\nAlready present; existing files will be skipped"
        options.append({**record, "selectable": skipped or configured or supplied, "detail": detail})
    return options


def run(args):
    targets = list(getattr(args, "targets", None) or [])
    if getattr(args, "all", False):
        if targets:
            raise ValueError("Use targets or --all")
        targets = list(TARGETS)
    unknown = set(targets) - set(TARGETS)
    if unknown:
        raise ValueError(f"Unknown download targets: {sorted(unknown)}")
    root = model_root(getattr(args, "model_dir", None))
    if getattr(args, "list", False):
        interactive = terminal_output()
        records = []
        for name in targets or TARGETS:
            status = resource_status(
                name,
                root,
                data_dir=getattr(args, "data_dir", None),
                config=getattr(args, "config", None),
            )
            status["url_configured"] = name in {"af2", "proteinmpnn", "solublempnn"}
            if interactive:
                records.append(status)
            else:
                print(json.dumps(status))
        if interactive:
            show_inventory(records, root)
        return 0
    if not targets:
        from partiflow.interactive import can_prompt, select_downloads

        if not can_prompt():
            raise ValueError("Provide download targets, --all, or --list")
        targets = select_downloads(_selection_options(args, root), root,
                                   importing=bool(getattr(args, "from_local", None)))
        if not targets:
            message("Cancelled")
            return 0
    if getattr(args, "url", None) and len(targets) != 1:
        raise ValueError("--url requires exactly one target")
    failed = False
    targets = list(dict.fromkeys(targets))
    with DownloadDisplay(root, targets, source=getattr(args, "from_local", None),
                         url=getattr(args, "url", None), force=getattr(args, "force", False)) as display:
        for target in targets:
            display.begin(target)
            try:
                result = acquire(
                    target,
                    root,
                    from_local=getattr(args, "from_local", None),
                    url=getattr(args, "url", None),
                    force=getattr(args, "force", False),
                    data_dir=getattr(args, "data_dir", None),
                    config=getattr(args, "config", None),
                )
            except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
                result = {
                    "target": target,
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            display.complete(result)
            if not display.active:
                print(json.dumps(result))
            failed |= result["status"] in {"failed", "unconfigured"}
    return int(failed)
