"""Shared by the step runners. Output goes to blueprint_data/_test/<step>/<version>/<pdf> (ignored by Git)."""
import atexit
import importlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
os.chdir(ROOT)
sys.path.append(str(ROOT))  # versions may import the app's main files
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
WORK = Path("blueprint_data/_test")
MAX_PAID_PAGES = 10


def versions(step: str) -> list[str]:
    return sorted(p.stem for p in (TESTS / step).glob("v*.py"))


def load_version(step: str, name: str):
    """Import tests/<step>/<name>.py."""
    if name not in versions(step):
        sys.exit(f"No version '{name}' in tests/{step}. Available: {', '.join(versions(step))}")
    sys.path.insert(0, str(TESTS / step))  # worker processes get this path too
    return importlib.import_module(name)


def out_dir(step: str, version: str) -> Path:
    return WORK / step / version


def pdf_of(pdf: str) -> Path:
    """The PDF in data/ whose name ends with v1 / v2."""
    found = sorted(Path("data").glob(f"*{pdf}.pdf"))
    if not found:
        sys.exit(f"No PDF ending with '{pdf}.pdf' in {ROOT / 'data'}")
    return found[0]


def load_json(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"{path} not found. Run the step before this one first.")
    return json.loads(path.read_text(encoding="utf-8"))


def confirm_cost(pages: int, allow_all: bool) -> None:
    if pages > MAX_PAID_PAGES and not allow_all:
        sys.exit(f"This would send {pages} pages to Gemini. Add --all if you really want that.")


class _Tee:
    """Writes to the screen and to a file."""

    def __init__(self, screen, file):
        self.screen, self.file = screen, file

    def write(self, text: str) -> int:
        self.screen.write(text)
        self.file.write(text)
        self.file.flush()  # nothing is lost if the run crashes
        return len(text)

    def flush(self) -> None:
        self.screen.flush()
        self.file.flush()

    def __getattr__(self, name: str):  # isatty, encoding, ...
        return getattr(self.screen, name)


def save_output(step: str, name: str) -> None:
    """From here on, everything printed (errors too) also goes to tests/<step>/outputs/<name>.txt."""
    folder = TESTS / step / "outputs"
    folder.mkdir(exist_ok=True)
    path = folder / (re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")[:100] + ".txt")
    file = path.open("w", encoding="utf-8")
    file.write(f"{datetime.now():%Y-%m-%d %H:%M:%S}  {' '.join(sys.argv)}")
    file.write(chr(10) * 2)
    sys.stdout, sys.stderr = _Tee(sys.stdout, file), _Tee(sys.stderr, file)
    atexit.register(lambda: sys.__stdout__.write(f"Saved to {path.relative_to(ROOT)}{chr(10)}"))
