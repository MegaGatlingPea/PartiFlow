"""Explicit downloads from the published model and data sources."""

from __future__ import annotations
import errno
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
HF_REPO = "MegaGatlingPea/PartiFlow"
HF_BASE = f"https://huggingface.co/{HF_REPO}/resolve/main"
PEPINTER_URL = f"{HF_BASE}/data/pepinter.tar.xz"
DEFAULT_TARGETS = ("autoencoder", "partiflow", "partiflow-ss", "partiflow-aux")


@dataclass(frozen=True)
class Asset:
    relative: str
    url: str | None = None


TARGETS = {
    name: (Asset(f"ckpts/{name}.ckpt", f"{HF_BASE}/ckpts/{name}.ckpt"),)
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
    ),
)
TARGETS["solublempnn"] = (
    Asset(
        "proteinmpnn/soluble/v_48_020.pt",
        "https://files.ipd.uw.edu/pub/ligandmpnn/solublempnn_v_48_020.pt",
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
            "archive": str(path / "pepinter.tar.xz"),
            "archive_present": (path / "pepinter.tar.xz").is_file()
            and (path / "pepinter.tar.xz").stat().st_size > 0,
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


def _find_asset(root, asset):
    for path in (root / asset.relative, root / Path(asset.relative).name):
        if path.is_file() and path.stat().st_size:
            return path
    matches = [path for path in root.rglob(Path(asset.relative).name)
               if path.is_file() and path.stat().st_size]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous downloaded asset: {asset.relative}")
    raise FileNotFoundError(f"Download has no {asset.relative}")


def _download_hub(parsed, destination):
    parts = parsed.path.strip("/").split("/")
    if (parsed.scheme != "https" or parsed.netloc != "huggingface.co"
            or len(parts) < 5 or parts[2] != "resolve"):
        return False
    from huggingface_hub import hf_hub_download

    repo = "/".join(parts[:2])
    revision = urllib.parse.unquote(parts[3])
    filename = urllib.parse.unquote("/".join(parts[4:]))
    try:
        fetched = hf_hub_download(
            repo_id=repo, filename=filename, revision=revision,
            endpoint="https://huggingface.co", token=False,
            local_dir=Path(destination).parent / "hub-download",
        )
    except Exception as exc:
        code = getattr(getattr(exc, "response", None), "status_code", None)
        detail = f"HTTP {code}" if code else type(exc).__name__
        raise OSError(f"Hugging Face download failed for {repo}/{filename}: {detail}") from None
    os.replace(fetched, destination)
    return True


def _fetch(url, destination):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"https", "http"}:
        raise ValueError("URL must use https or http")
    if _download_hub(parsed, destination):
        if not Path(destination).stat().st_size:
            raise ValueError("Hugging Face returned an empty file")
        return
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
    try:
        os.replace(source, destination)
        return
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
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


def _source_url(target):
    if target == "af2":
        return AF2_URL
    if target == "pepinter":
        return PEPINTER_URL
    return TARGETS[target][0].url


def download_status(target, root, *, data_dir=None, config=None):
    record = resource_status(target, root, data_dir=data_dir, config=config)
    if target == "pepinter":
        record = {
            **record, "data_dir": record["path"], "path": record["archive"],
            "training_data_ready": record["status"] == "present",
            "status": "present" if record["archive_present"] else "missing",
        }
        record["prepare_command"] = [
            "python", "-m", "partiflow.training.prepare_data",
            "--archive", record["archive"], "--root", record["data_dir"],
        ]
    return record


def acquire(target, root, *, url=None, force=False, data_dir=None, config=None):
    if target not in TARGETS:
        raise ValueError(f"Unknown download target: {target}")
    if url and urllib.parse.urlsplit(url).scheme not in {"http", "https"}:
        raise ValueError("URL must use https or http")
    root = Path(root).expanduser().resolve()
    existing = download_status(target, root, data_dir=data_dir, config=config)
    if existing["status"] == "present" and not force:
        return {**existing, "status": "skipped"}
    source_url = url or _source_url(target)
    if not source_url:
        raise ValueError(f"No download URL is configured for {target}")
    destination_root = data_root(root, data_dir=data_dir, config=config) if target == "pepinter" else root
    destination_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".partiflow-download-", dir=destination_root) as temporary:
        stage = Path(temporary)
        source_name = Path(urllib.parse.unquote(urllib.parse.urlsplit(source_url).path)).name or "download.bin"
        source = stage / ("download.tar" if target == "af2" else source_name)
        _fetch(source_url, source)
        if target == "pepinter":
            if not tarfile.is_tarfile(source):
                raise ValueError("PepInter download is not a readable structure archive")
            _publish_file(source, destination_root / "pepinter.tar.xz")
            return {**download_status(target, root, data_dir=destination_root), "status": "downloaded"}
        if _is_archive(source) or target == "af2":
            extracted = stage / "extracted"
            extract_archive(source, extracted)
            source = extracted
        pending = []
        for asset in TARGETS[target]:
            destination = root / asset.relative
            if destination.is_file() and destination.stat().st_size > 0 and not force:
                continue
            if source.is_file():
                if len(TARGETS[target]) != 1:
                    raise ValueError(f"{target} requires all configured assets in the downloaded archive")
                selected = source
            else:
                selected = _find_asset(source, asset)
            if not selected.stat().st_size:
                raise ValueError(f"Empty downloaded asset: {selected}")
            pending.append((selected, destination))
        for selected, destination in pending:
            _publish_file(selected, destination)
    return {**resource_status(target, root), "status": "downloaded"}


def _selection_options(args, root):
    """Read paths and configured sources without opening model files or the network."""
    options = []
    for name in TARGETS:
        record = download_status(name, root, data_dir=getattr(args, "data_dir", None),
                                 config=getattr(args, "config", None))
        source = "Explicit download URL" if getattr(args, "url", None) else _source_url(name)
        detail = RESOURCE_USAGE[name] + "\nStatus: " + record["status"] + "\nSource: " + source
        if record["status"] == "present" and not getattr(args, "force", False):
            detail += "\nAlready present; existing files will be skipped"
        if name == "pepinter":
            detail += "\nRaw archive; run data preprocessing before training"
        options.append({**record, "selectable": True, "detail": detail})
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
            status = download_status(
                name,
                root,
                data_dir=getattr(args, "data_dir", None),
                config=getattr(args, "config", None),
            )
            status["url_configured"] = bool(_source_url(name))
            status["url"] = _source_url(name)
            if interactive:
                records.append(status)
            else:
                print(json.dumps(status))
        if interactive:
            show_inventory(records, root)
        return 0
    if not targets:
        from partiflow.interactive import can_prompt, select_downloads

        if can_prompt():
            defaults = () if getattr(args, "url", None) else DEFAULT_TARGETS
            targets = select_downloads(
                _selection_options(args, root), root, default_values=defaults
            )
            if not targets:
                message("Cancelled")
                return 0
        else:
            if getattr(args, "url", None):
                raise ValueError("--url requires an explicit target outside a terminal")
            targets = list(DEFAULT_TARGETS)
    if getattr(args, "url", None) and len(targets) != 1:
        raise ValueError("--url requires exactly one target")
    failed = False
    targets = list(dict.fromkeys(targets))
    with DownloadDisplay(root, targets, url=getattr(args, "url", None),
                         force=getattr(args, "force", False)) as display:
        for target in targets:
            display.begin(target)
            try:
                result = acquire(
                    target,
                    root,
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
            failed |= result["status"] == "failed"
    return int(failed)
