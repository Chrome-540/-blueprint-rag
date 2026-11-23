"""Ask one question with one query version, on the output of an index version.

python tests/5_query/run.py v1_baseline "what is on sheet LS101?" --index v1_baseline --search-only
python tests/5_query/run.py v1_baseline "what is on sheet LS101?" --index v1_baseline

No automatic right/wrong: check the answer against the sheet yourself.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import load_version, out_dir, save_output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("version")
    ap.add_argument("question")
    ap.add_argument("--index", required=True, help="index version to search in")
    ap.add_argument("--search-only", action="store_true", help="only show which sheets are found")
    a = ap.parse_args()
    save_output("5_query", f"run_{a.version}_{a.question}" + ("_search" if a.search_only else ""))
    chroma = out_dir("4_index", a.index) / "chroma_db"
    if not chroma.exists():
        sys.exit(f"{chroma} not found. Run the index step first.")
    rag = load_version("5_query", a.version).BlueprintRAG(chroma_path=str(chroma), collection="test")
    start = time.time()
    answer, hits = ("", rag.hybrid_search(a.question, top_k=3)) if a.search_only else rag.ask(a.question)
    print("Sheets found:")
    for h in hits:
        m = h["metadata"]
        print(f"  {m['sheet_id']:<10} {m['file']} p.{m['page_number']}  {m['sheet_title'][:50]}")
    print(f"\n{answer}\n\n({time.time() - start:.1f}s)")


if __name__ == "__main__":
    main()
