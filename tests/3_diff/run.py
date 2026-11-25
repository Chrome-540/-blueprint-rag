"""Compare two sets. No Gemini. Run the ingest step for both PDFs first (whole PDFs).

python tests/3_diff/run.py v2_no_ai --ingest v3_position                        v1 -> v2
python tests/3_diff/run.py v2_no_ai --ingest v3_position --old v2 --new v1      v2 -> v1
python tests/3_diff/run.py v2_no_ai --ingest v3_position --show A101,A152       all sheets, details of these
"""
import argparse
import asyncio
import logging
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_json, load_version, out_dir, pdf_of, save_output

STEP = "3_diff"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("version")
    ap.add_argument("--ingest", required=True, help="ingest version to start from, e.g. v3_position")
    ap.add_argument("--old", choices=["v1", "v2"], default="v1")
    ap.add_argument("--new", choices=["v1", "v2"], default="v2")
    ap.add_argument("--show", default="", help="sheet IDs to print in detail, e.g. A101,A152")
    a = ap.parse_args()
    if a.version == "v1_baseline":
        sys.exit("v1_baseline sends every changed sheet to Gemini; it is only kept for reference.")
    if a.old == a.new:
        sys.exit("--old and --new must be different")
    save_output(STEP, f"run_{a.version}_{a.old}_to_{a.new}")
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    mod = load_version(STEP, a.version)
    sheets = {p: load_json(out_dir("1_ingest", a.ingest) / p / "manifest.json") for p in (a.old, a.new)}
    print(f"old: {pdf_of(a.old).name} ({len(sheets[a.old])} pages)\nnew: {pdf_of(a.new).name} ({len(sheets[a.new])} pages)\n")
    target = out_dir(STEP, a.version) / f"{a.old}_to_{a.new}"
    shutil.rmtree(target, ignore_errors=True)
    start = time.time()
    result = asyncio.run(mod.diff_versions_async(
        str(pdf_of(a.old)), str(pdf_of(a.new)), sheets[a.old], sheets[a.new], str(target), labels=(a.old, a.new)))
    secs = time.time() - start

    print(f"\n{'sheet':<14} {'status':<13} {'areas':>5}  {'added':>5} {'gone':>5} {'moved':>5}  title / problem")
    for s in result["sheets"]:
        n = [len(s["text"].get(k, [])) for k in ("added", "removed", "relocated")]
        print(f"{s['sheet_id']:<14} {s['status']:<13} {s['areas']:>5}  {n[0]:>5} {n[1]:>5} {n[2]:>5}  "
              f"{s['sheet_title'][:45]}{'  <-- ' + s['problem'] if s['problem'] else ''}")

    wanted = {x.strip().upper() for x in a.show.split(",") if x.strip()}
    for s in result["sheets"]:
        if s["sheet_id"] in wanted:
            print(f"\n===== {s['sheet_id']} {s['sheet_title']}: {s['status']}, {s['areas']} changed areas in the drawing")
            for where, texts in (("drawing", s["text"]), ("title block", s["title_text"])):
                for kind in ("added", "removed", "relocated"):
                    if texts.get(kind):
                        print(f"  {where}, text {kind}: {texts[kind]}")
            print(f"  picture: {s['overlay']}")
    print(f"\n{dict(Counter(s['status'] for s in result['sheets']))} in {secs:.0f}s")
    for s in result["sheets"]:
        assert s["status"] not in ("NEW_SHEET", "DELETED") or not s["overlay"]
        assert not s["overlay"] or Path(s["overlay"]).exists(), f"picture missing for {s['sheet_id']}"
    print(f"OK. Pictures are in {target / 'diffs'}")


if __name__ == "__main__":
    main()
