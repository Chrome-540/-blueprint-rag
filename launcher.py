"""Desktop launcher (adapted from Intellivault-chroma/launcher.py): starts the FastAPI backend +
Streamlit UI, opens the browser, adds a tray icon.

Dev run:   python launcher.py
Packaged:  build.bat (PyInstaller) then installer.iss (Inno Setup).
User data (API key, projects, blueprint_data incl. ChromaDB, app.log) lives in
%LOCALAPPDATA%/BlueprintRAG, never in the install dir.
"""
import logging
import multiprocessing
import os
import socket
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

BASE = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
USER_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "BlueprintRAG"
KEY_FILE = USER_DIR / "gemini_key.txt"
LOG_FILE = USER_DIR / "app.log"
API_PORT, UI_PORT = 8010, 8510  # IntelliVault uses 8000/8501; both apps can run side by side
UI_URL = f"http://localhost:{UI_PORT}"
os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"  # frozen app is otherwise seen as dev mode
os.environ["STREAMLIT_SERVER_PORT"] = str(UI_PORT)
os.environ["STREAMLIT_BROWSER_SERVER_PORT"] = str(UI_PORT)
os.environ["BACKEND_URL"] = f"http://127.0.0.1:{API_PORT}"


def port_free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def ask_key() -> str:
    import tkinter as tk
    from tkinter import simpledialog
    root = tk.Tk()
    root.withdraw()
    while True:
        key = simpledialog.askstring("Blueprint RAG setup", "Enter your Gemini API key:", parent=root)
        if key is None:
            sys.exit(0)
        if key.strip():
            return key.strip()


def setup_env() -> None:
    USER_DIR.mkdir(parents=True, exist_ok=True)
    if not KEY_FILE.exists():
        KEY_FILE.write_text(ask_key(), encoding="utf-8")
    os.environ.setdefault("GEMINI_API_KEY", KEY_FILE.read_text(encoding="utf-8").strip())
    os.chdir(USER_DIR)  # backend + engine use relative paths: projects/, blueprint_data/, app.log
    os.environ["IV_DATA_DIR"] = str(USER_DIR / "projects")
    sys.path[:0] = [str(BASE), str(BASE / "backend")]


def setup_logging() -> None:
    """One log file for everything: one line per stage event per file, one per chat."""
    handlers = [logging.FileHandler(LOG_FILE, encoding="utf-8")]
    if not getattr(sys, "frozen", False):
        handlers.append(logging.StreamHandler())  # dev: also echo to the console
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S", handlers=handlers, force=True)
    for noisy in ("httpx", "httpx2", "google_genai", "chromadb", "uvicorn", "uvicorn.access", "streamlit", "IntelliVaultUI"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def run_api() -> None:
    import uvicorn
    from main import app
    uvicorn.run(app, host="127.0.0.1", port=API_PORT, log_level="warning", log_config=None)


def open_when_ready() -> None:
    for _ in range(120):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{API_PORT}/domains", timeout=1)
            break
        except Exception:
            time.sleep(1)
    time.sleep(2)
    webbrowser.open(UI_URL)


def start_tray() -> None:
    import pystray
    from PIL import Image
    logo = BASE / "frontend" / "assets" / "logo.png"
    img = Image.open(logo) if logo.exists() else Image.new("RGB", (64, 64), "#2b6cb0")
    menu = pystray.Menu(
        pystray.MenuItem("Open Blueprint RAG", lambda: webbrowser.open(UI_URL), default=True),
        pystray.MenuItem("Open projects folder", lambda: os.startfile(USER_DIR / "projects")),
        pystray.MenuItem("Open log file", lambda: os.startfile(LOG_FILE)),
        pystray.MenuItem("Quit", lambda icon, item: (icon.stop(), os._exit(0))),
    )
    pystray.Icon("BlueprintRAG", img, "Blueprint RAG", menu).run_detached()


if __name__ == "__main__":
    multiprocessing.freeze_support()  # ingest worker processes re-launch this exe
    if not (port_free(API_PORT) and port_free(UI_PORT)):
        webbrowser.open(UI_URL)  # already running
        sys.exit(0)
    setup_env()
    if getattr(sys, "frozen", False):  # --windowed has no console; uvicorn/streamlit need real streams
        sys.stdout = sys.stderr = open(USER_DIR / "console.log", "a", encoding="utf-8", buffering=1)
    setup_logging()
    logging.getLogger("blueprint").info("=== Blueprint RAG starting ===")
    threading.Thread(target=run_api, daemon=True).start()
    threading.Thread(target=open_when_ready, daemon=True).start()
    start_tray()
    from streamlit.web import bootstrap
    flags = {
        "server.port": UI_PORT, "server.address": "localhost",  # never expose the key-holding app on the LAN
        "server.headless": True, "server.fileWatcherType": "none",
        "global.developmentMode": False, "browser.gatherUsageStats": False,
    }
    bootstrap.load_config_options(flags)  # bootstrap.run alone ignores flags -> dev mode, no static files
    bootstrap.run(str(BASE / "frontend" / "app.py"), False, [], flags)
