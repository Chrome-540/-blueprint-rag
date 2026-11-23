"""FastAPI backend (adapted from IntelliVault backend/main.py) on top of the blueprint engine.

Runs with cwd = user data dir (the launcher chdirs there):
  projects/<Project>/*.pdf               originals, one folder per project
  blueprint_data/<slug>/<file_slug>/...  rendered sheets + manifests (pipeline.py)
  blueprint_data/chroma_db               one Chroma, collection bp_<slug> per project
  app.log                                the log file (written by the launcher's logging setup)
"""
import asyncio
import importlib
import os
import re
import shutil
import sys
import threading
import time
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import quote

import chromadb
import pymupdf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from google import genai
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
import config  # noqa: E402
import pipeline  # noqa: E402
import router  # noqa: E402
from config import DATA_ROOT, DOMAINS, data_dir, slugify  # noqa: E402

BlueprintRAG = importlib.import_module("5_query_rag").BlueprintRAG
log = pipeline.log
LOG_FILE = Path("app.log")

app = FastAPI(title="Blueprint RAG API")
pipeline.BASE_DIR.mkdir(parents=True, exist_ok=True)
chroma = chromadb.PersistentClient(path=str(pipeline.CHROMA_PATH))
gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"].strip(),
                      http_options={"api_version": "v1beta", "timeout": 60_000})
log.info(f"Backend ready. Projects: {', '.join(DOMAINS) or '(none)'}")

UPLOAD_EXTS = {".pdf"}
NO_KB_MESSAGE = "No blueprints have been indexed for this project yet."

# ── Indexing ───────────────────────────────────────────────────────────────────────────────────
index_status: dict[str, dict] = {}  # slug -> {"state": indexing|done|error, "files": {file_slug: {...}}}
_project_locks: dict[str, threading.Lock] = defaultdict(threading.Lock)


def _pdfs(slug: str) -> list[Path]:
    return sorted(p for p in data_dir(slug).rglob("*") if p.is_file() and p.suffix.lower() in UPLOAD_EXTS)


def _index_project(slug: str) -> None:
    """Index every PDF of the project not indexed yet (pipeline skips finished files)."""
    entry = index_status.setdefault(slug, {"state": "indexing", "files": {}})
    with _project_locks[slug]:
        entry["state"] = "indexing"
        try:
            pdfs = _pdfs(slug)
            if pdfs:
                asyncio.run(pipeline.run(pdfs, entry["files"], slug))
                pair = pipeline.find_pair(pdfs)
                if pair and pipeline.load_diff(slug, pair) is None and all(
                        entry["files"][pipeline.slug(p)]["stage"].startswith("done") for p in pair):
                    entry["diff"] = {"file": "Compare v1 → v2", "stage": "queued", "done": 0, "total": None}
                    asyncio.run(pipeline.run_diff(slug, pair, entry["diff"]))
            stages = [f["stage"] for f in entry["files"].values()] + [entry.get("diff", {}).get("stage", "")]
            entry["state"] = "error" if any(s.startswith("error") for s in stages) else "done"
        except Exception as e:
            log.error(f"[{slug}] indexing crashed: {type(e).__name__}: {e}")
            entry["state"] = "error"


def _start_indexing(slug: str) -> None:
    config.refresh()
    index_status.setdefault(slug, {"state": "indexing", "files": {}})["state"] = "indexing"
    threading.Thread(target=_index_project, args=(slug,), daemon=True).start()


def _sync_all() -> None:
    for slug in list(DOMAINS.values()):
        _index_project(slug)


threading.Thread(target=_sync_all, daemon=True, name="project-sync").start()

# ── Retrieval (one BlueprintRAG per project, rebuilt when its sheet count changes) ──────────────
_rags: dict[str, tuple[int, object]] = {}


def _rag(slug: str):
    count = chroma.get_or_create_collection(pipeline.collection_name(slug), metadata={"hnsw:space": "cosine"}).count()
    cached = _rags.get(slug)
    if cached is None or cached[0] != count:  # new sheets landed -> reload BM25 corpus
        _rags[slug] = (count, BlueprintRAG(chroma_path=str(pipeline.CHROMA_PATH),
                                           collection=pipeline.collection_name(slug)))
    return _rags[slug][1]


def _sheet_card(slug: str, hit: dict) -> dict:
    m = hit["metadata"]
    file = m.get("file") or m["version"]
    filename = f"{m['sheet_id']}_{Path(file).stem}.pdf"
    return {
        "src": _data_url(m["image_path"]),
        "label": f"{m['sheet_id']} – {m['sheet_title']} · {file} (p.{m['page_number']})",
        "export": f"/sheet_pdf?project={quote(slug)}&file={quote(file)}&page={m['page_number']}&name={quote(filename)}",
        "filename": filename,
    }


def _data_url(path: str) -> str:
    return "/data/" + quote(Path(path).resolve().relative_to(pipeline.BASE_DIR.resolve()).as_posix())


# ── Differences between v1 and v2 (stored by 3_diff_versions.py after indexing) ─────────────────
_STATUS_TEXT = {"MODIFIED": "Changed", "NEW_SHEET": "Added", "DELETED": "Deleted", "NEEDS_REVIEW": "Check by hand"}
_DIFF_NOTE = "*Differences are computed from the PDFs, not written by AI. Very small changes may not be detected; text drawn as lines shows only in the picture.*"


def _load_diff(slug: str) -> tuple[dict | None, tuple | None, str]:
    """(stored comparison, (v1, v2) PDFs, message when there is nothing to show)."""
    pair = pipeline.find_pair(_pdfs(slug))
    if pair is None:
        return None, None, ("There is nothing to compare in this project. I need two PDFs with the same name, "
                            "one containing v1 and one v2 (e.g. 'Set v1.pdf' and 'Set v2.pdf').")
    if data := pipeline.load_diff(slug, pair):
        return data, pair, ""
    entry = index_status.get(slug, {})
    if entry.get("state") == "indexing":
        return None, pair, "v1 and v2 are still being indexed or compared. Please ask again when that finishes."
    if entry.get("diff", {}).get("stage", "").startswith("error"):
        return None, pair, "The comparison of v1 and v2 failed. Check the log file, then restart the app to try again."
    return None, pair, "v1 and v2 have not been compared yet. Restart the app to run the comparison."


def _cell(text: str) -> str:
    return re.sub(r"\s+", " ", text).replace("|", "/").strip()


def _change_text(s: dict) -> str:
    """One table cell, facts only: '3 text added, 1 removed, drawing changed (4 areas)'."""
    if s["status"] in ("NEW_SHEET", "DELETED"):
        return "Only in v2" if s["status"] == "NEW_SHEET" else "Only in v1"
    if s["problem"]:
        return "Not compared: " + s["problem"]
    names = {"added": "text added", "removed": "text removed", "relocated": "text in a new place"}
    parts = [f"{len(s['text'][k])} {name}" for k, name in names.items() if s["text"][k]]
    if s["areas"]:
        parts.append(f"drawing changed ({s['areas']} area{'s' if s['areas'] > 1 else ''})")
    return ", ".join(parts)


def _text_block(title: str, texts: list[str]) -> str:
    """Every text exactly as printed on the sheet (in code marks, so Markdown can't alter it)."""
    items = [f"`{t.replace('`', chr(39))}`" + (f" ×{n}" if n > 1 else "") for t, n in Counter(texts).items()]
    return f"**{title}:** " + " · ".join(items) if items else ""


def _show_differences(slug: str, ids: list[str], desc: str) -> dict:
    data, pair, problem = _load_diff(slug)
    if data is None:
        return {"answer": problem, "images": [], "pdfs": []}

    if not ids and not desc:  # whole set: counts + table, no images
        rows = [s for s in data["sheets"] if s["status"] in _STATUS_TEXT]
        n = Counter(s["status"] for s in data["sheets"])
        parts = [f"**Differences from v1 to v2** ({pair[0].name} → {pair[1].name})",
                 f"{n['MODIFIED']} sheets changed, {n['NEW_SHEET']} added, {n['DELETED']} deleted, "
                 f"{n['NEEDS_REVIEW']} to check by hand. {n['TITLE_ONLY']} changed only in the title block or stamp, "
                 f"{n['UNCHANGED']} show no differences."]
        if rows:
            parts.append("| Sheet | Title | Status | Change |\n|---|---|---|---|\n" + "\n".join(
                f"| {_cell(s['sheet_id'])} | {_cell(s['sheet_title'])} | {_STATUS_TEXT[s['status']]} | {_cell(_change_text(s))} |"
                for s in rows))
            parts.append(f'To see one sheet, ask e.g. "show me the differences in {rows[0]["sheet_id"]}".')
        return {"answer": "\n\n".join(parts + [_DIFF_NOTE]), "images": [], "pdfs": []}

    items = [{"diff": s, "metadata": {"sheet_id": s["sheet_id"], "sheet_title": s["sheet_title"],
                                      "page_number": (s["curr"] or s["prev"])[0]["page_number"]}} for s in data["sheets"]]
    found = [i["diff"] for i in router.find_sheets(items, ids, desc)][:3]
    if not found:
        return {"answer": f"I couldn't find sheet {', '.join(ids) or desc} in v1 or v2.", "images": [], "pdfs": []}
    intros = {"UNCHANGED": "no differences found between v1 and v2.", "TITLE_ONLY": "only the title block or stamp changed.",
              "NEW_SHEET": "added in v2; it does not exist in v1.", "DELETED": "deleted; it exists only in v1.",
              "MODIFIED": "differences from v1 to v2."}
    texts, images = [], []
    for s in found:
        intro = intros.get(s["status"]) or f"not compared, {s['problem']}. Please check the sheets below by hand."
        block = [f"**{s['sheet_id']} – {s['sheet_title']}**: {intro}"]
        for key, where in (("text", ""), ("title_text", "Title block: ")):
            for kind, name in (("added", "text added"), ("removed", "text removed"), ("relocated", "same text, new place")):
                block.append(_text_block((where + name).capitalize() if not where else where + name, s[key].get(kind, [])))
        if s["areas"]:
            block.append(f"**Drawing changed** in {s['areas']} area{'s' if s['areas'] > 1 else ''}: see red and green in the picture.")
        texts.append("\n\n".join(x for x in block if x))
        if s["overlay"]:
            images.append({
                "src": _data_url(s["overlay"]),
                "label": f"{s['sheet_id']} differences: red = only in v1, green = only in v2, gray = unchanged",
                "export": f"/diff_png?project={quote(slug)}&sheet={quote(s['sheet_id'])}",
                "filename": Path(s["overlay"]).name})
        for side, pdf in (("prev", pair[0]), ("curr", pair[1])):
            for loc in s[side]:
                images.append(_sheet_card(slug, {"metadata": {
                    **loc, "sheet_id": s["sheet_id"], "sheet_title": s["sheet_title"], "file": pdf.name}}))
    return {"answer": "\n\n".join(texts + [_DIFF_NOTE]), "images": images, "pdfs": []}


# ── Chat ───────────────────────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    domain: str
    message: str
    history: list[dict] = []


@app.get("/domains")
def list_domains() -> list[str]:
    return list(DOMAINS)


@app.post("/chat")
def chat(req: ChatRequest) -> dict:
    slug = DOMAINS.get(req.domain)
    if slug is None:
        raise HTTPException(status_code=400, detail=f"Unknown project: {req.domain}")
    t0 = time.time()
    intent = router.detect_intent(gemini, req.message, req.history, req.domain)
    name, args = intent["name"], intent["args"]
    try:
        result = _dispatch(slug, req, name, args)
    except Exception as e:
        log.error(f'[chat] {req.domain}: "{req.message[:80]}" -> {name} failed: {type(e).__name__}: {e}')
        raise
    shown = ", ".join(str(v) for v in args.values() if v and name not in ("answer_from_context", "small_talk"))
    log.info(f'[chat] {req.domain}: "{req.message[:80]}" -> {name}({shown}) -> '
             f'{len(result["images"])} sheets, {len(result["pdfs"])} pdfs ({time.time() - t0:.1f}s)')
    return result


def _dispatch(slug: str, req: ChatRequest, name: str, args: dict) -> dict:
    images, pdfs = [], []

    if name == "small_talk":  # reply was written by the router call itself: no search, no 2nd LLM call
        return {"answer": args.get("reply") or "Hi! Ask me about the drawings, or say 'show A101'.",
                "images": [], "pdfs": []}

    if name == "show_file":
        folder = data_dir(slug)
        rels = [p.relative_to(folder).as_posix() for p in _pdfs(slug)]
        matched = router.match_names(args.get("file_hint", ""), rels) or (rels if len(rels) == 1 else [])
        if not matched:
            listing = "\n".join(f"- {r}" for r in rels) or "(no files)"
            return {"answer": f"I couldn't tell which file you mean. Files in this project:\n\n{listing}",
                    "images": [], "pdfs": []}
        pdfs = [f"/projects/{quote(folder.name)}/{quote(r)}" for r in matched[:3]]
        return {"answer": "Here's the file you requested:" if len(pdfs) == 1 else "Here are the matching files:",
                "images": [], "pdfs": pdfs}

    if name == "show_differences":
        return _show_differences(slug, list(args.get("sheet_ids") or []), args.get("description", ""))

    rag = _rag(slug)
    if not rag.corpus:
        return {"answer": NO_KB_MESSAGE, "images": [], "pdfs": []}

    if name == "show_sheet":
        ids, desc = list(args.get("sheet_ids") or []), args.get("description", "")
        hits = router.find_sheets(rag.corpus, ids, desc)
        if not hits and desc:
            hits = rag.hybrid_search(desc, top_k=2)
        if not hits:
            return {"answer": f"I couldn't find sheet {', '.join(ids) or desc} in this project.",
                    "images": [], "pdfs": []}
        shown = hits[: router.MAX_SHEET_CARDS]
        answer = "Here is the sheet:" if len(shown) == 1 else f"Found {len(hits)} matching sheets:"
        if len(hits) > len(shown):
            answer += f" (showing {len(shown)}; also: " + ", ".join(
                f"{h['metadata']['sheet_id']} ({h['metadata'].get('file', '')})" for h in hits[len(shown):]) + ")"
        return {"answer": answer, "images": [_sheet_card(slug, h) for h in shown], "pdfs": []}

    if name == "list_contents":
        return {"answer": router.list_contents(rag.corpus, args.get("file_hint", ""), args.get("discipline", "")),
                "images": [], "pdfs": []}

    answer, hits = rag.ask(args.get("query") or req.message, history=req.history)
    if hits:
        images.append(_sheet_card(slug, hits[0]))  # top cited sheet, like IV shows the top image
    return {"answer": answer, "images": images, "pdfs": pdfs}


# ── Files ──────────────────────────────────────────────────────────────────────────────────────
# Only these two folders are served; never the user dir root (it holds gemini_key.txt).
DATA_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/data", StaticFiles(directory=pipeline.BASE_DIR), name="data")
app.mount("/projects", StaticFiles(directory=DATA_ROOT), name="projects")


@app.get("/sheet_pdf")
def sheet_pdf(project: str, file: str, page: int, name: str = "sheet.pdf") -> Response:
    """One page of a project PDF as its own vector PDF (the sheet card's Export -> download)."""
    if project not in DOMAINS.values():
        raise HTTPException(status_code=404, detail="Unknown project")
    src_path = next((p for p in _pdfs(project) if p.name == file), None)
    if src_path is None:
        raise HTTPException(status_code=404, detail="File not found")
    with pymupdf.open(src_path) as src, pymupdf.open() as dst:
        if not 1 <= page <= src.page_count:
            raise HTTPException(status_code=404, detail="Page out of range")
        dst.insert_pdf(src, from_page=page - 1, to_page=page - 1)
        data = dst.tobytes()
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name) or "sheet.pdf"
    return Response(data, media_type="application/pdf",
                    headers={"Content-Disposition": f'attachment; filename="{safe}"'})


@app.get("/diff_png")
def diff_png(project: str, sheet: str) -> FileResponse:
    """Red/green overlay as a download. The path comes from the stored comparison, never the client."""
    if project not in DOMAINS.values():
        raise HTTPException(status_code=404, detail="Unknown project")
    data = _load_diff(project)[0] or {"sheets": []}
    path = next((s["overlay"] for s in data["sheets"] if s["sheet_id"] == sheet), None)
    if not path or not Path(path).is_file():
        raise HTTPException(status_code=404, detail="No differences image for this sheet")
    return FileResponse(path, filename=Path(path).name)


@app.get("/logs")
def logs(n: int = 15) -> dict:
    if not LOG_FILE.exists():
        return {"lines": []}
    with LOG_FILE.open(encoding="utf-8", errors="replace") as f:
        return {"lines": [line.rstrip() for line in f.readlines()[-min(n, 200):]]}


# ── Upload / projects (adapted from IntelliVault) ──────────────────────────────────────────────
def _forget_file(slug: str, name: str) -> None:
    """A replaced PDF: drop its vectors, rendered sheets and status so it is re-indexed."""
    file_slug = pipeline.slug(name)
    pair = pipeline.find_pair(_pdfs(slug))
    if pair and file_slug in map(pipeline.slug, pair):  # the stored comparison is out of date
        pipeline.diff_summary_path(slug, pair).unlink(missing_ok=True)
    try:
        chroma.get_collection(pipeline.collection_name(slug)).delete(where={"version": file_slug})
    except Exception:
        pass  # collection not created yet
    shutil.rmtree(pipeline.BASE_DIR / slug / file_slug, ignore_errors=True)
    index_status.get(slug, {}).get("files", {}).pop(file_slug, None)


@app.post("/upload")
def upload(domain: str = Form(...), files: list[UploadFile] = File(...)) -> dict:
    """Save PDFs into a project folder (created if new) and index them in the background."""
    label = _clean_label(domain)
    folder = DATA_ROOT / (next((l for l in DOMAINS if l.lower() == label.lower()), label))
    folder.mkdir(parents=True, exist_ok=True)
    slug = slugify(folder.name)  # indexing of new files waits for any run already in progress

    saved, skipped = [], []
    for f in files:
        name = Path(f.filename or "").name
        if Path(name).suffix.lower() not in UPLOAD_EXTS:
            skipped.append(name)
            continue
        dest = folder / name
        if dest.exists():
            _forget_file(slug, name)
        with dest.open("wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(name)

    if saved:
        _start_indexing(slug)
    log.info(f"[upload] {folder.name}: saved {len(saved)} file(s)" + (f", skipped {skipped}" if skipped else ""))
    return {"domain": folder.name, "saved": saved, "skipped": skipped}


def _clean_label(name: str) -> str:
    label = re.sub(r'[<>:"/\|?*]+', " ", name).strip().strip(".")
    if not slugify(label):
        raise HTTPException(status_code=400, detail="Invalid project name")
    return label


def _extract_zip(zf: zipfile.ZipFile, dest: Path) -> int:
    """Extract supported files keeping subfolders. Drops a single wrapping top folder; rejects zip-slip."""
    members = [m for m in zf.infolist()
               if not m.is_dir() and not m.filename.startswith("__MACOSX/")
               and Path(m.filename).suffix.lower() in UPLOAD_EXTS]
    parts = [Path(m.filename.replace("\\", "/")).parts for m in members]
    if any(".." in p or Path(m.filename).is_absolute() for m, p in zip(members, parts)):
        raise ValueError("Unsafe paths in zip")
    strip = 1 if parts and all(len(p) > 1 and p[0] == parts[0][0] for p in parts) else 0
    root = dest.resolve()
    for m, p in zip(members, parts):
        target = (dest / Path(*p[strip:])).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"Unsafe path in zip: {m.filename}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(m) as src, open(target, "wb") as out:
            shutil.copyfileobj(src, out)
    return len(members)


def _copy_folder(src: Path, dest: Path) -> int:
    """Copy supported files from a local folder, keeping subfolders."""
    count = 0
    for f in src.rglob("*"):
        if f.is_file() and f.suffix.lower() in UPLOAD_EXTS:
            target = dest / f.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, target)
            count += 1
    return count


@app.post("/add_project")
def add_project(name: str = Form(...), zip_file: UploadFile | None = File(None), path: str = Form("")) -> dict:
    """Create a new project from a ZIP or a local folder path (subfolders kept), then index it."""
    label = _clean_label(name)
    if any(l.lower() == label.lower() for l in DOMAINS) or slugify(label) in DOMAINS.values():
        raise HTTPException(status_code=409, detail=f"Project '{label}' already exists, use Add files instead")
    if (zip_file is None) == (not path.strip()):
        raise HTTPException(status_code=400, detail="Provide either a ZIP file or a folder path")
    src = Path(path.strip().strip('"')) if path.strip() else None
    if src is not None and not src.is_dir():
        raise HTTPException(status_code=400, detail=f"Folder not found: {src}")

    folder = DATA_ROOT / label
    folder.mkdir(parents=True)
    try:
        if src is not None:
            count = _copy_folder(src, folder)
        else:
            with zipfile.ZipFile(zip_file.file) as zf:
                count = _extract_zip(zf, folder)
        if not count:
            raise ValueError("No PDFs found")
    except (ValueError, zipfile.BadZipFile, OSError) as e:
        shutil.rmtree(folder, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(e))

    _start_indexing(slugify(label))
    log.info(f"[add_project] {label}: {count} PDF(s) from {src or 'zip'}")
    return {"domain": label, "saved": count}


@app.delete("/domains/{domain}")
def delete_domain(domain: str) -> dict:
    """Drop the project's Chroma collection, rendered sheets and original PDFs."""
    slug = DOMAINS.get(domain)
    if slug is None:
        raise HTTPException(status_code=404, detail=f"Unknown project: {domain}")
    lock = _project_locks[slug]
    if not lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Indexing in progress, try again shortly")
    try:
        try:
            chroma.delete_collection(pipeline.collection_name(slug))
        except Exception:
            pass  # never indexed
        _rags.pop(slug, None)
        shutil.rmtree(pipeline.BASE_DIR / slug, ignore_errors=True)
        shutil.rmtree(data_dir(slug))
        config.refresh()
        index_status.pop(slug, None)
    finally:
        lock.release()
    log.info(f"[delete] project {domain} removed")
    return {"deleted": domain}


@app.get("/index_status")
def get_index_status() -> dict:
    return index_status
