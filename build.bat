@echo off
REM Run inside the activated venv:  uv pip install -r requirements-build.txt  then  build.bat
REM Close any running BlueprintRAG.exe first (tray -> Quit), or the dist folder is locked.
python -c "from PIL import Image; Image.open('frontend/assets/logo.png').save('icon.ico', sizes=[(256,256),(64,64),(32,32)])"
pyinstaller --noconfirm --clean --onedir --windowed --name BlueprintRAG --icon icon.ico ^
  --paths . --paths backend ^
  --collect-all streamlit --collect-all chromadb --collect-all pymupdf ^
  --copy-metadata streamlit --copy-metadata chromadb --copy-metadata google-genai ^
  --copy-metadata fastapi --copy-metadata uvicorn --copy-metadata pymupdf ^
  --hidden-import main --hidden-import config --hidden-import router --hidden-import pipeline ^
  --hidden-import 1_ingest_pdf --hidden-import 2_enrich_async --hidden-import 3_diff_versions --hidden-import cv2 --hidden-import 4_index_chroma --hidden-import 5_query_rag ^
  --hidden-import python_multipart --hidden-import multipart --hidden-import fitz --hidden-import rank_bm25 ^
  --hidden-import pystray._win32 --hidden-import tkinter.simpledialog --hidden-import tkinter.filedialog ^
  --exclude-module torch --exclude-module sentence_transformers ^
  --add-data "frontend;frontend" ^
  launcher.py
REM Then compile installer.iss with Inno Setup (iscc installer.iss) -> Output\BlueprintRAG-Setup.exe
