"""Show every page where two ingest versions differ. Run both with run.py first.

python tests/1_ingest/compare.py v1_baseline v3_position v1
"""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_json, out_dir, save_output

FIELDS = ("sheet_id", "sheet_title", "revision_date", "raw_cad_text")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("pdf", choices=["v1", "v2"])
    a = ap.parse_args()
    save_output("1_ingest", f"compare_{a.old}_{a.new}_{a.pdf}")
    old, new = (load_json(out_dir("1_ingest", v) / a.pdf / "manifest.json") for v in (a.old, a.new))
    if len(old) != len(new):
        sys.exit(f"Different number of pages: {len(old)} and {len(new)}. Run both with the same --pages.")
    counts = dict.fromkeys([*FIELDS, "image"], 0)
    for x, y in zip(old, new):
        for f in FIELDS:
            if x[f] != y[f]:
                counts[f] += 1
                if f != "raw_cad_text":
                    print(f"p{x['page_number']:<4} {f:<14} {str(x[f])[:45]!r:<48} -> {str(y[f])[:45]!r}")
        if len({hashlib.md5(Path(r["image_path"]).read_bytes()).digest() for r in (x, y)}) == 2:
            counts["image"] += 1
            print(f"p{x['page_number']:<4} image differs")
    print("\nPages that differ:", counts)


if __name__ == "__main__":
    main()
