"""Project registry (copied from IntelliVault's domain registry), auto-discovered from DATA_ROOT.

Every subfolder of DATA_ROOT ("projects") is a project = one blueprint set. The folder name is the
label shown in the UI; its slug (lowercase, non-alphanumerics -> "_") names its Chroma collection
(bp_<slug>) and its blueprint_data/<slug> folder.
"""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.getenv("IV_DATA_DIR", "projects"))  # relative to cwd = user data dir


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


_DIRS: dict[str, Path] = {}  # slug -> folder
DOMAINS: dict[str, str] = {}  # label -> slug  (mutated in place so importers see new domains)


def refresh() -> None:
    """Re-scan the data folder; call after adding a domain folder at runtime."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    _DIRS.clear()
    DOMAINS.clear()
    for p in sorted(DATA_ROOT.iterdir()):
        if p.is_dir() and not p.name.startswith((".", "_")) and slugify(p.name):
            _DIRS[slugify(p.name)] = p
            DOMAINS[p.name] = slugify(p.name)


refresh()


def data_dir(slug: str) -> Path:
    return _DIRS[slug]
