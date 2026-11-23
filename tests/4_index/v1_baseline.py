import json
import os
from pathlib import Path
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()
from google.genai import types


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
        f"Raw CAD Text:\n{sheet.get('raw_cad_text', '')[:8000]}"
    )


def embed_texts_batched(client: genai.Client, texts: list[str], batch_size: int = 50) -> list[list[float]]:
    all_embeddings: list[list[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        res = client.models.embed_content(
            model="gemini-embedding-001",
            contents=batch,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768,
            ),
        )
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
        http_options={"api_version": "v1beta"},
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
            }

            ids.append(doc_id)
            documents.append(doc_text)
            metadatas.append(meta)

        embeddings = embed_texts_batched(client, documents, batch_size=50)
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