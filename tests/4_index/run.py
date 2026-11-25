"""Run one index version on the output of an enrich version. Uses Gemini embeddings (very cheap).

python tests/4_index/run.py v1_baseline v1 --enrich v1_baseline
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_json, load_version, out_dir, save_output

import chromadb

STEP = "4_index"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("version")
    ap.add_argument("pdf", choices=["v1", "v2"])
    ap.add_argument("--enrich", required=True, help="enrich version to start from")
    a = ap.parse_args()
    save_output(STEP, f"run_{a.version}_{a.pdf}_enrich_{a.enrich}")
    source = out_dir("2_enrich", a.enrich)
    records = load_json(source / a.pdf / "enriched_manifest.json")
    mod, chroma = load_version(STEP, a.version), str(out_dir(STEP, a.version) / "chroma_db")
    start = time.time()
    n = mod.index_versions_hybrid([a.pdf], base_dir=str(source), chroma_path=chroma, collection="test",
                                  file_names={a.pdf: f"{a.pdf}.pdf"})
    got = chromadb.PersistentClient(path=chroma).get_collection("test").get(where={"version": a.pdf})
    for m in sorted(got["metadatas"], key=lambda m: m["page_number"]):
        print(f"{m['page_number']:>4}  {m['sheet_id']:<14} {m['discipline']:<16} {m['sheet_title'][:50]}")
    assert n == len(records), "not every sheet was indexed"
    assert {m["page_number"] for m in got["metadatas"]} >= {r["page_number"] for r in records}, "sheets missing in ChromaDB"
    print(f"\nOK: {n} sheets indexed in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
