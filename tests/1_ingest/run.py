"""Run one ingest version. No Gemini.

python tests/1_ingest/run.py v3_position v1              whole PDF
python tests/1_ingest/run.py v3_position v1 --pages 5    first 5 pages
"""
import argparse
import asyncio
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_json, load_version, out_dir, pdf_of, save_output

import pymupdf

STEP, CHUNK_PAGES = "1_ingest", 4


async def ingest_parallel(mod, pdf: str, name: str, out: Path, pool, total: int) -> None:
    """Same steps as _ingest in _pipeline_parallel.py."""
    loop, pages = asyncio.get_running_loop(), list(range(1, total + 1))
    img_dir = out / name / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    starts = range(0, total, CHUNK_PAGES)
    chunks = await asyncio.gather(*(
        loop.run_in_executor(pool, mod.read_pages, pdf, pages[i:i + CHUNK_PAGES]) for i in starts))
    records = mod.build_records([t for c in chunks for t in c], name, img_dir)
    jobs = [(r.page_number, r.image_path) for r in records]
    await asyncio.gather(*(
        loop.run_in_executor(pool, mod.render_pages, pdf, jobs[i:i + CHUNK_PAGES]) for i in starts))
    mod.save_manifest(records, out / name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("version")
    ap.add_argument("pdf", choices=["v1", "v2"])
    ap.add_argument("--pages", type=int, help="only the first N pages")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    save_output(STEP, f"run_{a.version}_{a.pdf}" + (f"_{a.pages}pages" if a.pages else ""))
    mod, out = load_version(STEP, a.version), out_dir(STEP, a.version)

    shutil.rmtree(out / a.pdf, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    copy = out / f"{a.pdf}.pdf"
    with pymupdf.open(pdf_of(a.pdf)) as src, pymupdf.open() as dst:
        n = min(a.pages or src.page_count, src.page_count)
        dst.insert_pdf(src, from_page=0, to_page=n - 1)
        dst.save(copy)

    locked = False
    start = time.time()
    if hasattr(mod, "read_pages"):
        with ProcessPoolExecutor(a.workers) as pool:
            asyncio.run(ingest_parallel(mod, str(copy), a.pdf, out, pool, n))
            secs = time.time() - start
            try:  # the workers are still alive here: the PDF must not be locked
                os.remove(copy)
            except PermissionError:
                locked = True
    else:
        mod.ingest_blueprint_pdf(str(copy), a.pdf, output_dir=str(out))
        secs = time.time() - start

    records = load_json(out / a.pdf / "manifest.json")
    for r in records:
        print(f"{r['page_number']:>4}  {r['sheet_id']:<14} {r['sheet_title'][:50]:<50} "
              f"{r['revision_date']}  {' '.join(r.get('revisions', []))}")
    images = list((out / a.pdf / "images").iterdir())
    assert [r["page_number"] for r in records] == list(range(1, n + 1)), "pages missing or out of order"
    assert len({r["canonical_id"] for r in records}) == n, "sheet names are not unique"
    assert len(images) == n and all(p.suffix == ".png" for p in images), "images missing or half-written"
    for r in records:
        with pymupdf.open(r["image_path"]) as im:  # fails on a broken image
            assert im.page_count == 1
    print(f"\n{a.version}: {n} pages in {secs:.1f}s, "
          f"{sum(r['sheet_id'].startswith('UNKNOWN') for r in records)} without a sheet ID, "
          f"{sum(r['sheet_title'] == 'UNTITLED SHEET' for r in records)} without a title")
    assert not locked, "a worker still holds the PDF"
    print("OK: all images readable" + (", PDF not locked" if hasattr(mod, "read_pages") else ""))


if __name__ == "__main__":
    main()
