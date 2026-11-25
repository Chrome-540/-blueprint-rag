"""Ask every question in questions.json and check the answers. 1 Gemini call + 1 embedding per question.

python tests/5_query/check.py v3_full_text --index v2_full_text

Three automatic checks: the right sheet was found, the expected words are in the answer, and every number, code and
name in the answer is printed on the sheets found (groundedness).
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import load_version, out_dir, save_output

NOT_FOUND = re.compile(
    r"\b(not|no|none|cannot|can't|unable|doesn't|don't|isn't|aren't|without)\b", re.IGNORECASE)


def has(word: str, answer: str) -> bool:
    """`5` must not be found inside `25` or `1.5`."""
    clean = re.sub(r"\s+", " ", answer.lower().replace("*", ""))
    return re.search(rf"(?<![\w.]){re.escape(word.lower())}(?!\d|\.\d)", clean) is not None


def plain(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def ungrounded(answer: str, hits: list[dict], question: str) -> list[str]:
    """Numbers, codes and CAPITALIZED names of the answer that are not printed on the sheets found.
    Uses only the PDF's own text (not Gemini's summary). Names read from a picture are flagged too."""
    printed, known = "", question
    for h in hits:
        printed += " " + h["text"].split("Raw CAD Text:", 1)[-1]
        m = h["metadata"]
        known += f" {m['sheet_id']} {m['sheet_title']} {m['file']} {m['page_number']} {m.get('discipline', '')}"
    words = {plain(w) for w in re.split(r"[\s,;:()\[\]]+", printed + " " + known)}
    joined = plain(printed + known)
    text = re.sub(r"[*`]", "", answer)
    text = re.sub(r"(?m)^\s*\d+[.)]\s", "", text)  # numbers of a list
    items = re.findall(r"[A-Za-z]*\d[\w./'\"-]*|\b[A-Z][A-Z/&.-]{2,}(?:\s+[A-Z][A-Z/&.-]+)*\b", text)
    bad = []
    for item in dict.fromkeys(i.strip(".,;:'\"-") for i in items):
        p = plain(item)
        # numbers and short items must be a whole word; longer names may be split over several words ("MERV-13")
        if p and not (p in words or (len(p) >= 4 and not p.isdigit() and p in joined)):
            bad.append(item)
    return bad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("version")
    ap.add_argument("--index", required=True)
    a = ap.parse_args()
    save_output("5_query", f"check_{a.version}_index_{a.index}")
    chroma = out_dir("4_index", a.index) / "chroma_db"
    if not chroma.exists():
        sys.exit(f"{chroma} not found. Run the index step first.")
    rag = load_version("5_query", a.version).BlueprintRAG(chroma_path=str(chroma), collection="test")
    questions = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))

    results, times, clean = [], [], 0
    for i, t in enumerate(questions, 1):
        start = time.time()
        try:
            answer, hits = rag.ask(t["q"])
        except Exception as e:
            answer, hits = f"CALL FAILED {type(e).__name__}: {str(e)[:150]}", []
        times.append(time.time() - start)
        found = [h["metadata"]["sheet_id"] for h in hits]
        problems = []
        if answer.startswith("CALL FAILED"):
            problems.append("call failed")
        elif t.get("not_in_drawings"):
            if not NOT_FOUND.search(answer):
                problems.append("should say it is not in the drawings")
        else:
            if t["sheet"] not in found:
                problems.append(f"sheet {t['sheet']} not found")
            problems += [f"missing '{w}'" for w in t["must"] if not has(w, answer)]
            problems += [f"must not say '{w}'" for w in t.get("must_not", []) if has(w, answer)]
        results.append(not problems)
        bad = ungrounded(answer, hits, t["q"])
        clean += not bad
        print(f"\n{i:>2}. {'ok   ' if not problems else 'CHECK'} {t['q']}")
        print(f"    expected: {'NOT IN DRAWINGS' if t.get('not_in_drawings') else ', '.join(t['must']) + ' on ' + t['sheet']}"
              f"   ({t['source']})")
        print(f"    sheets found: {', '.join(found)}   {times[-1]:.1f}s")
        print("    answer: " + " ".join(answer.split()))
        if bad:
            print(f"    NOT PRINTED ON THESE SHEETS: {', '.join(bad)}")
        if problems:
            print("    PROBLEM: " + "; ".join(problems))
    print(f"\n{sum(results)} of {len(results)} correct, {clean} of {len(results)} fully grounded, {sum(times) / len(times):.1f}s per question, slowest {max(times):.1f}s")


if __name__ == "__main__":
    main()
