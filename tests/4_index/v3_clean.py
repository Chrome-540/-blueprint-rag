import json
import logging
import os
import time
from collections import Counter
from pathlib import Path
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()
from google.genai import errors, types

log = logging.getLogger("blueprint")


def build_searchable_document(sheet: dict) -> str:
    """Combines structured VLM metadata, version diffs, and raw CAD text into a rich retrieval document."""
    diff_info = sheet.get("diff_from_prev") or {}
    diff_block = ""
    if diff_info:
        diff_block = (
            f"\nVersion Diff Status ({diff_info.get('compared_against')} -> {sheet['version']}): "
            f"{diff_info.get('status')}\n"
            f"Revision Changes: {diff_info.get('change_summary', '')}\n"
        )

    return (
        f"Version: {sheet['version']} | Sheet ID: {sheet['sheet_id']} ({sheet['canonical_id']})\n"
        f"Title: {sheet['sheet_title']}\n"
        f"Discipline: {sheet.get('discipline', 'Unknown')} | Floor/Scope: {sheet.get('floor_level', 'Unknown')}\n"
        f"Grid Bounds: {sheet.get('grid_bounds', 'N/A')}\n"
        f"Referenced Sheets: {', '.join(sheet.get('referenced_sheets', []))}\n"
        f"Key Entities: {', '.join(sheet.get('key_entities', []))}\n"
        f"Spatial Summary: {sheet.get('spatial_summary', '')}\n"
        f"{diff_block}"
        f"Raw CAD Text:\n{sheet.get('raw_cad_text', '')}"
    )


def _lines(text: str) -> list[str]:
    return [" ".join(line.split()) for line in text.splitlines() if line.strip()]


def repeated_lines(sheets: list[dict]) -> set[str]:
    """Lines printed on more than half of the sheets: general notes, copyright, title block labels."""
    if len(sheets) < 10:  # too few sheets to tell what is repeated
        return set()
    counts = Counter(line for s in sheets for line in set(_lines(s.get("raw_cad_text", ""))))
    return {line for line, n in counts.items() if n > len(sheets) / 2}


def vector_text(sheet: dict, repeated: set[str]) -> str:
    """Text for the vector only: without the repeated lines, with the names seen only in the image."""
    own = "\n".join(line for line in _lines(sheet.get("raw_cad_text", "")) if line not in repeated)
    extra = ", ".join(sheet.get("unconfirmed_entities", []))
    return (f"Seen in the image: {extra}\n" if extra else "") + build_searchable_document({**sheet, "raw_cad_text": own})


def _embed(client: genai.Client, batch: list[str], max_retries: int = 5):
    """One embedding call, repeated after network errors and rate limits."""
    for attempt in range(1, max_retries + 1):
        try:
            return client.models.embed_content(
                model="gemini-embedding-001",
                contents=batch,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT", output_dimensionality=768),
            )
        except Exception as e:
            fatal = isinstance(e, errors.APIError) and e.code != 429 and 400 <= e.code < 500  # bad key or request
            if fatal or attempt == max_retries:
                raise
            log.warning(f"embedding retry {attempt}/{max_retries} in {2 ** attempt}s: {type(e).__name__}: {str(e)[:120]}")
            time.sleep(2 ** attempt)


def embed_texts_batched(client: genai.Client, texts: list[str], batch_size: int = 50) -> list[list[float]]:
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        res = _embed(client, batch)
        all_embeddings.extend([e.values for e in res.embeddings])
    return all_embeddings


def index_versions_hybrid(
    versions: list[str] = ["v1", "v2"],
    base_dir: str = "./blueprint_data",
    chroma_path: str | None = None,
    collection: str = "blueprint_sheets",
    file_names: dict[str, str] | None = None,
) -> int:
    """Embed + upsert each version's enriched sheets. `file_names` maps version -> original PDF
    name (shown in the UI). Defaults keep the old single-project CLI layout."""
    base = Path(base_dir)
    index_dir = base / "index"
    chroma_path = chroma_path or str(index_dir / "chroma_db")
    Path(chroma_path).mkdir(parents=True, exist_ok=True)

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"].strip(),
        http_options={"api_version": "v1beta", "timeout": 60_000},
    )
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    collection = chroma_client.get_or_create_collection(
        name=collection,
        metadata={"hnsw:space": "cosine"},
    )
    file_names = file_names or {}

    total = 0
    for ver in versions:
        manifest_path = base / ver / "enriched_manifest.json"
        sheets = json.loads(manifest_path.read_text(encoding="utf-8"))

        ids: list[str] = []
        documents: list[str] = []
        vector_texts: list[str] = []
        repeated = repeated_lines(sheets)
        metadatas: list[dict] = []

        for s in sheets:
            doc_id = f"{ver}__{s['canonical_id']}"
            doc_text = build_searchable_document(s)
            diff_info = s.get("diff_from_prev") or {}

            meta = {
                "version": ver,
                "file": file_names.get(ver, ver),
                "page_number": int(s["page_number"]),
                "sheet_id": s["sheet_id"],
                "canonical_id": s["canonical_id"],
                "sheet_title": s["sheet_title"],
                "discipline": s.get("discipline", "Unknown"),
                "floor_level": s.get("floor_level", "Unknown"),
                "image_path": s["image_path"],
                "referenced_sheets": json.dumps(s.get("referenced_sheets", [])),
                "diff_status": diff_info.get("status", "BASE_VERSION"),
                "change_summary": diff_info.get("change_summary", ""),
                "diff_overlay_path": diff_info.get("diff_overlay_path") or "",
                "search_extra": " ".join(s.get("unconfirmed_entities", [])),  # search only, never sent to Gemini
            }

            ids.append(doc_id)
            documents.append(doc_text)
            vector_texts.append(vector_text(s, repeated))
            metadatas.append(meta)

        # The FULL text is stored and searched by keyword. Only the vector uses the start of it (as before):
        # the embedding model reads a limited amount of text anyway.
        embeddings = embed_texts_batched(client, [d[:9000] for d in vector_texts], batch_size=50)
        collection.delete(where={"version": ver})  # re-index: no leftovers under old sheet IDs
        collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        total += len(ids)

    # BM25 is rebuilt from the Chroma collection at query time (5_query_rag.py), so indexing
    # one file at a time never overwrites the others.
    return total


if __name__ == "__main__":
    print(f"Indexed {index_versions_hybrid(versions=['v1', 'v2'])} sheets.")