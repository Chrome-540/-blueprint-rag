import importlib
import json
import os
import re
from pathlib import Path
import chromadb
from PIL import Image
from rank_bm25 import BM25Okapi
from google import genai
from dotenv import load_dotenv

load_dotenv()
from google.genai import types

enrich = importlib.import_module("2_enrich_async")  # the model is defined there


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_.-]+", text.lower())


class BlueprintRAG:
    def __init__(self, base_dir: str = "./blueprint_data", chroma_path: str | None = None,
                 collection: str = "blueprint_sheets"):
        self.base = Path(base_dir)
        self.index_dir = self.base / "index"
        self.client = genai.Client(
            api_key=os.environ["GEMINI_API_KEY"].strip(),
            http_options={"api_version": "v1beta", "timeout": 120_000},
        )

        chroma_client = chromadb.PersistentClient(path=chroma_path or str(self.index_dir / "chroma_db"))
        self.collection = chroma_client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )

        # BM25 corpus comes straight from Chroma: one source of truth, always matches the vectors.
        # ponytail: whole corpus in RAM, fine for ~10k sheets; page through collection.get beyond that.
        got = self.collection.get(include=["documents", "metadatas"])
        self.corpus = [
            {"doc_id": i, "text": d, "metadata": m}
            for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])
        ]
        self.by_doc_id = {item["doc_id"]: item for item in self.corpus}
        self.bm25 = BM25Okapi([tokenize(item["text"]) for item in self.corpus]) if self.corpus else None

    def _detect_query_mode(self, query: str) -> tuple[str, bool]:
        q = query.lower()
        is_diff = any(
            w in q
            for w in [
                "change",
                "changed",
                "diff",
                "revision",
                "modified",
                "v1 to v2",
                "between v1",
                "updated",
                "deleted",
                "removed",
            ]
        )
        if "v1" in q and not is_diff:
            return "v1", False
        return "v2", is_diff

    def hybrid_search(self, query: str, version: str | None = None, top_k: int = 4) -> list[dict]:
        """version=None searches every indexed file; otherwise only that file/version."""
        if not self.corpus:
            return []
        in_scope = lambda meta: version is None or meta["version"] == version  # noqa: E731
        # 1. Semantic Search (ChromaDB, optionally filtered by version)
        q_emb = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=query,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
                output_dimensionality=768,
            ),
        ).embeddings[0].values

        sem_res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=top_k * 3,
            where={"version": version} if version else None,
        )
        sem_ids = sem_res["ids"][0] if sem_res["ids"] else []

        # 2. Lexical Search (BM25 filtered by version)
        scores = self.bm25.get_scores(tokenize(query))
        bm25_ranked = [
            self.corpus[idx]["doc_id"]
            for idx in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
            if in_scope(self.corpus[idx]["metadata"]) and scores[idx] > 0
        ][: top_k * 3]

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_scores: dict[str, float] = {}
        k_rrf = 60
        for rank, doc_id in enumerate(sem_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank + 1)
        for rank, doc_id in enumerate(bm25_ranked):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank + 1)

        # Boost exact Sheet ID mentions in query (e.g., "A152", "E801", "S202")
        mentioned_ids = set(re.findall(r"\b[A-Z]{1,3}\d{2,3}[A-Z]?\b", query.upper()))
        for doc_id, item in self.by_doc_id.items():
            if in_scope(item["metadata"]) and item["metadata"]["sheet_id"] in mentioned_ids:
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0

        # Skip ids indexed after this instance loaded its corpus (files still being indexed).
        top_doc_ids = [d for d in sorted(rrf_scores, key=rrf_scores.get, reverse=True) if d in self.by_doc_id]
        return [self.by_doc_id[did] for did in top_doc_ids[:top_k]]

    def ask(self, query: str, top_k: int = 3, history: list[dict] | None = None) -> tuple[str, list[dict]]:
        """Returns (answer, retrieved hits). `history` = [{role, content}], last 6 used for follow-ups."""
        # Multi-file mode: search every indexed file. The v1/v2 routing in _detect_query_mode is
        # parked with the diff step; only its diff-intent flag is used here.
        _, is_diff = self._detect_query_mode(query)
        hits = self.hybrid_search(query, top_k=top_k)
        if not hits:
            return "No indexed blueprints yet. Upload PDFs and index them first.", []

        # Include global v1->v2 summary context if user asks a broad revision question
        global_diff_context = ""
        if is_diff:
            summary_file = self.base / "v2" / "diff_summary_v1_to_v2.json"
            if summary_file.exists():
                summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
                deleted = summary_data.get("deleted_sheets", [])
                modified = [m["sheet_id"] for m in summary_data.get("modified_sheets", [])]
                global_diff_context = (
                    f"\nGlobal v1 -> v2 Summary: {len(modified)} sheets modified ({', '.join(modified[:30])}...), "
                    f"Deleted from v1: {json.dumps(deleted)}.\n"
                )

        contents: list = []
        context_blocks: list[str] = [global_diff_context] if global_diff_context else []

        for hit in hits:
            meta = hit["metadata"]
            context_blocks.append(
                f"--- SHEET {meta['sheet_id']} (File {meta.get('file', meta['version'])}, Page {meta['page_number']}) ---\n"
                f"Title: {meta['sheet_title']} | Discipline: {meta['discipline']} | Scope: {meta['floor_level']}\n"
                f"Diff Status: {meta['diff_status']} | Change Summary: {meta['change_summary']}\n"
                f"{hit['text'][:4000]}"
            )

            img_path = Path(meta["image_path"])
            if img_path.exists():
                with Image.open(img_path) as im:
                    im.load()
                    contents.append(im.copy())

            # Attach visual red/green diff overlay when answering revision questions
            overlay_str = meta.get("diff_overlay_path")
            if is_diff and overlay_str and Path(overlay_str).exists():
                with Image.open(overlay_str) as ov:
                    ov.load()
                    contents.append(ov.copy())

        history_block = ""
        if history:
            lines = [f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}" for m in history[-6:]]
            history_block = "\nConversation History:\n" + "\n".join(lines) + "\n"

        prompt = f"""
You are a senior construction & architectural AI assistant answering questions over stamped blueprints.
Use BOTH the attached blueprint images (and red/green diff overlays where attached) and the retrieved CAD/metadata context.
Always cite exact Sheet IDs, File, and Page Numbers for every claim.

Retrieved Context:
{chr(10).join(context_blocks)}
{history_block}
User Question: {query}
"""
        contents.append(prompt)

        response = self.client.models.generate_content(
            model=enrich.GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.1),
        )
        return (response.text or "").strip(), hits


if __name__ == "__main__":
    rag = BlueprintRAG()
    while True:
        q = input("\nAsk a blueprint question (or 'exit'): ").strip()
        if not q or q.lower() == "exit":
            break
        print("\n" + rag.ask(q)[0])