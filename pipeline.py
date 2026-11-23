"""Pipelined multi-file indexing: every PDF runs ingest -> enrich -> index as its own chain, so
file B renders (CPU, process pool) while file A is being enriched (Gemini, shared semaphore).

Layout: blueprint_data/<project>/<file_slug>/...   +   blueprint_data/chroma_db (collection bp_<project>)

CLI:  python pipeline.py <folder-or-pdfs...> [--project NAME]
      python pipeline.py --selftest <one.pdf>
Knobs (env): INGEST_WORKERS (default 8), GEMINI_CONCURRENCY (default 100), VLM_MAX_SIDE, GEMINI_MODEL.
"""
import asyncio
import importlib
import logging
import multiprocessing
import os
import re
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pymupdf
from dotenv import load_dotenv
from google import genai

load_dotenv()
ingest_mod = importlib.import_module("1_ingest_pdf")
enrich_mod = importlib.import_module("2_enrich_async")
index_mod = importlib.import_module("4_index_chroma")

log = logging.getLogger("blueprint")
BASE_DIR = Path("./blueprint_data")
CHROMA_PATH = BASE_DIR / "chroma_db"
LOG_EVERY = 50  # pages between progress lines in the log
_CHROMA_WRITE = threading.Lock()  # one Chroma writer per process, across concurrent project runs


def slug(name: str | Path) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", Path(name).stem).strip("_").lower()


def collection_name(project: str) -> str:
    return f"bp_{project}"  # Chroma names need >= 3 chars


def _ingest(pdf: str, name: str, out_dir: str) -> int:
    """Runs in a worker process; returns only the page count (records stay on disk)."""
    return len(ingest_mod.ingest_blueprint_pdf(pdf_path=pdf, version=name, output_dir=out_dir))


def _index(out_dir: Path, name: str, project: str, file_name: str) -> int:
    with _CHROMA_WRITE:
        return index_mod.index_versions_hybrid(
            [name], base_dir=str(out_dir), chroma_path=str(CHROMA_PATH),
            collection=collection_name(project), file_names={name: file_name},
        )


class _Progress:
    """Counts finished pages into the UI status dict and logs every LOG_EVERY pages."""

    def __init__(self, st: dict, tag: str, stage: str):
        self.st, self.tag, self.stage, self.logged = st, tag, stage, 0

    def set(self, done: int) -> None:
        self.st["done"] = done
        if done // LOG_EVERY > self.logged and done < (self.st["total"] or 0):
            self.logged = done // LOG_EVERY
            log.info(f"{self.tag}: {self.stage} {done}/{self.st['total']}")

    def inc(self) -> None:
        self.set(self.st["done"] + 1)


async def process_file(pdf, project, out_dir, pool, sem, client, status):
    name, file_name = slug(pdf), Path(pdf).name
    st, tag = status[name], f"[{project}] {file_name}"
    t0 = time.perf_counter()
    stage = "ingest"
    done_marker = out_dir / name / ".indexed"
    try:
        if done_marker.exists():
            st.update(stage="done (already indexed)")
            return
        loop = asyncio.get_running_loop()
        with pymupdf.open(pdf) as doc:
            total = doc.page_count
        st.update(stage="ingesting", done=0, total=total, t0=time.time())
        log.info(f"{tag}: ingest started ({total} pages)")
        # Worker process can't touch `status`; it writes one PNG per page, so count those instead.
        img_dir, prog = out_dir / name / "images", _Progress(st, tag, "ingest")
        fut = loop.run_in_executor(pool, _ingest, str(pdf), name, str(out_dir))
        while not fut.done():
            if img_dir.exists():
                prog.set(min(total, sum(1 for _ in img_dir.glob("*.png"))))
            await asyncio.wait({fut}, timeout=1)
        st["pages"] = fut.result()
        log.info(f"{tag}: ingest done ({st['pages']} pages, {time.perf_counter() - t0:.0f}s)")

        stage, t1 = "enrich", time.perf_counter()
        st.update(stage="enriching", done=0, total=st["pages"])
        log.info(f"{tag}: enrich started")
        prog = _Progress(st, tag, "enrich")
        records = await enrich_mod.enrich_manifest_async(
            version=name, output_dir=str(out_dir), semaphore=sem, client=client, on_progress=prog.inc, tag=tag,
        )
        skipped = sum(1 for r in records if r.get("enrich_error"))
        log.info(f"{tag}: enrich done ({len(records) - skipped} ok, {skipped} text-only, {time.perf_counter() - t1:.0f}s)")

        stage, t2 = "index", time.perf_counter()
        st.update(stage="indexing")
        n = await asyncio.to_thread(_index, out_dir, name, project, file_name)
        log.info(f"{tag}: indexed {n} sheets ({time.perf_counter() - t2:.0f}s)")
        if skipped:  # no marker: the next run re-enriches only those pages, then re-indexes
            st.update(stage=f"done ({skipped} pages text-only, re-run to retry)")
        else:
            done_marker.touch()
            st.update(stage="done")
    except Exception as e:  # one bad PDF must not kill the batch
        log.error(f"{tag}: failed at {stage}: {type(e).__name__}: {str(e)[:200]}")
        st.update(stage=f"error: {type(e).__name__}: {e}"[:300])
    finally:
        st["secs"] = round(time.perf_counter() - t0, 1)


async def run(pdfs: list, status: dict, project: str = "default") -> dict:
    """Index `pdfs` of one project concurrently. `status` is filled live:
    {file_slug: {file, stage, pages, secs, done, total, t0}}."""
    out_dir = BASE_DIR / project
    for p in pdfs:
        status.setdefault(slug(p), {"file": Path(p).name, "stage": "queued", "pages": None, "secs": None,
                                    "done": 0, "total": None, "t0": None})
    sem = asyncio.Semaphore(int(os.getenv("GEMINI_CONCURRENCY", "100")))
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"].strip(), http_options=enrich_mod.HTTP_OPTIONS)
    # ponytail: ~100 MB raw pixmap per 200-DPI page per worker; lower INGEST_WORKERS on low-RAM machines.
    workers = max(1, min(len(pdfs), int(os.getenv("INGEST_WORKERS", "8"))))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        await asyncio.gather(*(process_file(p, project, out_dir, pool, sem, client, status) for p in pdfs))
    return status


def collect_pdfs(args: list[str]) -> list[Path]:
    out: list[Path] = []
    for a in args:
        p = Path(a)
        out += sorted(p.glob("*.pdf")) if p.is_dir() else [p]
    return out


def _selftest(pdf: str) -> None:
    """Index the first 3 pages of `pdf` into a throwaway dir and check they are searchable."""
    import shutil
    import tempfile

    tmp = Path(tempfile.mkdtemp())
    small = tmp / "selftest.pdf"
    with pymupdf.open(pdf) as src, pymupdf.open() as dst:
        dst.insert_pdf(src, from_page=0, to_page=2)
        dst.save(small)
    cwd = os.getcwd()
    os.chdir(tmp)  # every stage works relative to ./blueprint_data
    try:
        status = asyncio.run(run([small], {}, project="selftest"))
        st = status["selftest"]
        assert st["stage"] == "done", st
        assert st["done"] == st["total"] == 3, st  # progress counters reached the end
        rag = importlib.import_module("5_query_rag").BlueprintRAG(
            chroma_path=str(CHROMA_PATH), collection=collection_name("selftest"))
        assert rag.collection.count() == 3, rag.collection.count()
        assert rag.hybrid_search("sheet", top_k=2), "no hits"
        assert {h["metadata"]["file"] for h in rag.corpus} == {"selftest.pdf"}
        print("SELFTEST OK", st)
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "google_genai", "chromadb"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = sys.argv[1:]
    if args[:1] == ["--selftest"]:
        _selftest(args[1])
        sys.exit(0)
    project = "default"
    if "--project" in args:
        i = args.index("--project")
        project, args = slug(args[i + 1]), args[:i] + args[i + 2:]
    pdfs = collect_pdfs(args)
    if not pdfs:
        sys.exit("usage: python pipeline.py <folder-or-pdfs...> [--project NAME]")
    t = time.perf_counter()
    for name, s in asyncio.run(run(pdfs, {}, project)).items():
        print(f"{s['stage']:<12} {s['pages'] or '-':>4} pages {s['secs']:>7}s  {s['file']}")
    print(f"Total wall time: {time.perf_counter() - t:.1f}s")
