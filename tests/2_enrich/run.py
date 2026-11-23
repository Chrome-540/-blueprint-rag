"""Run one enrich version on the output of an ingest version. COSTS MONEY: 1 Gemini call per page.

python tests/2_enrich/run.py v2_highres v1 --ingest v3_position --only 2,29,50,122,150
"""
import argparse
import asyncio
import inspect
import json
import logging
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import confirm_cost, load_json, load_version, out_dir, save_output

STEP = "2_enrich"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("version")
    ap.add_argument("pdf", choices=["v1", "v2"])
    ap.add_argument("--ingest", required=True, help="ingest version to start from, e.g. v3_position")
    ap.add_argument("--only", help="page numbers, e.g. 2,29,50")
    ap.add_argument("--all", action="store_true", help="allow more than 10 pages")
    ap.add_argument("--concurrency", type=int, default=5)
    a = ap.parse_args()
    save_output(STEP, f"run_{a.version}_{a.pdf}_ingest_{a.ingest}" + (f"_pages_{a.only}" if a.only else ""))
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    before = load_json(out_dir("1_ingest", a.ingest) / a.pdf / "manifest.json")
    all_ids = {r["sheet_id"] for r in before}
    if a.only:
        wanted = {int(x) for x in a.only.split(",")}
        before = [r for r in before if r["page_number"] in wanted]
    confirm_cost(len(before), a.all)
    mod, out = load_version(STEP, a.version), out_dir(STEP, a.version)
    target = out / a.pdf
    shutil.rmtree(target, ignore_errors=True)
    (target / "images").mkdir(parents=True)
    records = []
    for r in before:  # work on copies: enrich renames images
        image = target / "images" / Path(r["image_path"]).name
        shutil.copy2(r["image_path"], image)
        records.append({**r, "image_path": str(image)})
    (target / "manifest.json").write_text(json.dumps(records), encoding="utf-8")

    start = time.time()
    extra = {"known_ids": all_ids} if "known_ids" in inspect.signature(mod.enrich_manifest_async).parameters else {}
    records = asyncio.run(mod.enrich_manifest_async(a.pdf, output_dir=str(out), max_concurrency=a.concurrency, **extra))
    secs = time.time() - start
    print(f"model: {getattr(mod, 'GEMINI_MODEL', '?')}\npage  ingest         Gemini read    used")
    guess = {r["page_number"]: r["sheet_id"] for r in before}
    for r in records:
        read = r.get("verified_sheet_id", "-")
        flag = "  <-- Gemini disagrees" if read != r["sheet_id"] else ""
        print(f"{r['page_number']:>4}  {guess[r['page_number']]:<14} {read:<14} {r['sheet_id']:<14}{flag}")
        print(f"      {r.get('discipline', '-')} | {r['sheet_title'][:60]} | floor: {r.get('floor_level', '-')}")
        print(f"      SUMMARY: {r.get('spatial_summary', '-')}")
        label = "CONFIRMED NAMES" if "unconfirmed_entities" in r else "NAMES (not checked in this version)"
        print(f"      {label}: {', '.join(r.get('key_entities', []))}")
        print(f"      UNCONFIRMED (only seen in the image): {', '.join(r.get('unconfirmed_entities', []))}")
        print(f"      IMAGE TOKENS: {r.get('image_tokens', 'not recorded')}")
        print(f"      REFERENCES: {', '.join(r.get('referenced_sheets', []))}")
    failed = [r["page_number"] for r in records if r.get("enrich_error")]
    print(f"\n{a.version}: {len(records) - len(failed)} of {len(records)} pages enriched in {secs:.1f}s, tokens: {getattr(mod, 'USAGE', 'not recorded')}")
    assert not failed, f"pages that failed: {failed}"
    print("OK")


if __name__ == "__main__":
    main()
