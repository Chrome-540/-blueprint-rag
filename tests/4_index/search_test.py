"""How often does search find the right sheet? All sheets of one PDF, from ingest only (no Enrich, no answers).
Uses embeddings only (a few cents at most). Run it for two index versions and compare the numbers.

python tests/4_index/search_test.py v2_full_text
python tests/4_index/search_test.py v3_clean
python tests/4_index/search_test.py v2_full_text --reuse     don't embed the sheets again
python tests/4_index/search_test.py v2_full_text --reuse --query v5_fusion     test that query version itself

It compares ways to combine keyword and meaning search on the same questions:
  now       RRF with 60, 9 results per list, 3 sheets (what the app does)
  A         keep the #1 of each list, RRF for the rest
  B         RRF with 5 instead of 60
  C         30 results per list instead of 9
  D         5 sheets instead of 3
The questions are made by code, always the same ones:
  title   "show me the roof framing plan"           -> the sheet(s) with that title
  note    a line printed on one sheet only          -> that sheet
  deep    the same, but from after character 9,000  -> that sheet
"""
import argparse
import importlib
import json
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import TESTS, load_json, load_version, out_dir, save_output

PER_KIND = 40


def make_questions(records: list[dict]) -> list[tuple[str, str, set[str]]]:
    """(kind, question, right sheets)"""
    rnd, out = random.Random(1), []
    by_title = defaultdict(set)
    for r in records:
        if r["sheet_title"] != "UNTITLED SHEET":
            by_title[r["sheet_title"]].add(r["sheet_id"])
    titles = sorted(by_title)
    for t in rnd.sample(titles, min(PER_KIND, len(titles))):
        out.append(("title", f"show me the {t.lower()}", by_title[t]))

    def clean(line: str) -> str:
        return " ".join(line.split())

    counts = Counter(x for r in records for x in {clean(l) for l in r["raw_cad_text"].splitlines()})

    def good(line: str) -> bool:
        return counts[line] == 1 and 25 <= len(line) <= 90 and len(re.findall(r"[A-Za-z]{4,}", line)) >= 4

    notes, deep = [], []
    for r in records:
        pos, early, late = 0, [], []
        for raw in r["raw_cad_text"].splitlines(keepends=True):
            line = clean(raw)
            if good(line):
                (late if pos > 9000 else early).append(line)
            pos += len(raw)
        if early:
            notes.append(("note", rnd.choice(early).lower(), {r["sheet_id"]}))
        if late:
            deep.append(("deep", rnd.choice(late).lower(), {r["sheet_id"]}))
    for group in (notes, deep):
        out += rnd.sample(group, min(PER_KIND, len(group)))
    return out


def fuse(keyword: list[str], meaning: list[str], boosted: set[str], k: int = 60, depth: int = 9, top: int = 3,
         keep_first: bool = False) -> list[str]:
    """RRF over two ranked lists of document ids. `boosted`: documents whose sheet ID is in the question."""
    score: dict[str, float] = {}
    for ranked in (meaning[:depth], keyword[:depth]):
        for place, doc in enumerate(ranked):
            score[doc] = score.get(doc, 0.0) + 1.0 / (k + place + 1)
    for doc in boosted:
        score[doc] = score.get(doc, 0.0) + 1.0
    ranked = sorted(score, key=score.get, reverse=True)
    if keep_first and not boosted:
        ranked = list(dict.fromkeys(keyword[:1] + meaning[:1] + ranked))
    return ranked[:top]


METHODS = {
    "keyword only": lambda kw, me, bo: kw[:3],
    "meaning only": lambda kw, me, bo: me[:3],
    "now": lambda kw, me, bo: fuse(kw, me, bo),
    "A keep #1s": lambda kw, me, bo: fuse(kw, me, bo, keep_first=True),
    "B RRF 5": lambda kw, me, bo: fuse(kw, me, bo, k=5),
    "C 30 results": lambda kw, me, bo: fuse(kw, me, bo, depth=30),
    "D 5 sheets": lambda kw, me, bo: fuse(kw, me, bo, depth=15, top=5),
    "A + C": lambda kw, me, bo: fuse(kw, me, bo, depth=30, keep_first=True),
    "A + D": lambda kw, me, bo: fuse(kw, me, bo, depth=15, top=5, keep_first=True),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("version", help="index version")
    ap.add_argument("--query", default="v4_search_extra", help="query version used for searching")
    ap.add_argument("--ingest", default="v3_position")
    ap.add_argument("--pdf", choices=["v1", "v2"], default="v1")
    ap.add_argument("--reuse", action="store_true", help="use the index of the last run")
    a = ap.parse_args()
    save_output("4_index", f"search_{a.version}_{a.pdf}")

    records = load_json(out_dir("1_ingest", a.ingest) / a.pdf / "manifest.json")
    if len(records) < 100:
        sys.exit(f"Only {len(records)} pages in the ingest output. Run ingest on the whole PDF first.")
    base = out_dir("4_index", a.version + "_search")
    if not (a.reuse and (base / "chroma_db").exists()):
        shutil.rmtree(base, ignore_errors=True)
        (base / a.pdf).mkdir(parents=True)
        (base / a.pdf / "enriched_manifest.json").write_text(json.dumps(records), encoding="utf-8")
        start = time.time()
        n = load_version("4_index", a.version).index_versions_hybrid(
            [a.pdf], base_dir=str(base), chroma_path=str(base / "chroma_db"), collection="search")
        print(f"{a.version}: {n} sheets indexed in {time.time() - start:.0f}s")
        assert n == len(records)

    sys.path.insert(0, str(TESTS / "5_query"))
    query = importlib.import_module(a.query)
    rag = query.BlueprintRAG(chroma_path=str(base / "chroma_db"), collection="search")
    assert len(rag.corpus) == len(records), "the stored index does not match. Run without --reuse."
    questions = make_questions(records)
    from google.genai import types
    vectors = []
    for i in range(0, len(questions), 50):  # all questions in a few calls instead of one call each
        vectors += [e.values for e in rag.client.models.embed_content(
            model="gemini-embedding-001", contents=[q for _, q, _ in questions[i:i + 50]],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY", output_dimensionality=768)).embeddings]

    sheet_of = {d["doc_id"]: d["metadata"]["sheet_id"] for d in rag.corpus}
    top_k, itself = getattr(query, "TOP_K", 3), f"** {a.query}"
    hit = {m: Counter() for m in [*METHODS, itself]}
    total, different, misses = Counter(), 0, defaultdict(list)
    for (kind, q, right), vector in zip(questions, vectors):
        scores = rag.bm25.get_scores(query.tokenize(q))
        keyword = [rag.corpus[i]["doc_id"] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True) if scores[i] > 0]
        meaning = rag.collection.query(query_embeddings=[vector], n_results=30)["ids"][0]
        named = set(re.findall(r"\b[A-Z]{1,3}\d{2,3}[A-Z]?\b", q.upper()))
        boosted = {d for d, s in sheet_of.items() if s in named}

        # "now" must be exactly what the app's own search returns
        answer = SimpleNamespace(embeddings=[SimpleNamespace(values=vector)])
        rag.client = SimpleNamespace(models=SimpleNamespace(embed_content=lambda **k: answer))
        app = [h["metadata"]["sheet_id"] for h in rag.hybrid_search(q, top_k=top_k)]
        different += set(app) != {sheet_of[d] for d in METHODS["now" if top_k == 3 else "A + D"](keyword, meaning, boosted)}

        total[kind] += 1
        total["ALL"] += 1
        for name in hit:
            found = app if name == itself else [sheet_of[d] for d in METHODS[name](keyword, meaning, boosted)]
            if right & set(found):
                hit[name][kind] += 1
                hit[name]["ALL"] += 1
            else:
                misses[name].append(f"[{kind}] {q!r}: wanted {', '.join(sorted(right))}, found {', '.join(found)}")

    kinds = ("title", "note", "deep", "ALL")
    print(f"\nRight sheet found:{'':<4}" + "".join(f"{k + ' (' + str(total[k]) + ')':>14}" for k in kinds))
    for name in hit:
        print(f"  {name:<20}" + "".join(f"{100 * hit[name][k] / max(1, total[k]):>13.0f}%" for k in kinds))
    best = max((m for m in METHODS if m.startswith(("A", "B", "C"))), key=lambda m: hit[m]["ALL"])
    print(f"\nStill missed with '{best}':")
    for line in misses[best]:
        print("  " + line)
    print(f"\nQuestions where '{'now' if top_k == 3 else 'A + D'}' differs from {a.query}'s own search: {different} (should be 0, or a few ties)")


if __name__ == "__main__":
    main()
