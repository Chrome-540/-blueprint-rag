



"""
IntelliVault RAG Pipeline (v2)
────────────────────────────────
PDF/Image → Hierarchical Chunking + Vision Captioning → Gemini Embeddings → ChromaDB → Gemini 2.5 Flash
Uses: google-genai (new SDK), chromadb, langchain-text-splitters, pdfplumber, PyMuPDF
"""

from __future__ import annotations
import importlib, re, time, logging, random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

log = logging.getLogger("IntelliVaultRAG")

_query_log = logging.getLogger("IntelliVaultRAG.queries")
_query_log.setLevel(logging.INFO)
_query_log.propagate = False
_log_file = Path("query_log.txt")  # in the user data folder: the install folder is not writable
_fh = logging.FileHandler(_log_file, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(message)s"))
_query_log.addHandler(_fh)

import fitz  # PyMuPDF — used to render scanned/visual pages
import pdfplumber
from google import genai
from google.genai import types
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# ── Section detection ──────────────────────────────────────────────────────────
SECTION_PATTERNS = [
    (r"(?i)table\s*of\s*contents",        "Table of Contents"),
    (r"(?i)introduction\b",               "Introduction"),
    (r"(?i)overview\b",                   "Overview"),
    (r"(?i)standard\s*operating\s*procedure|sop\b", "Standard Operating Procedure"),
    (r"(?i)troubleshoot(ing)?",           "Troubleshooting"),
    (r"(?i)safety\s*(procedures?|guidelines?)", "Safety"),
    (r"(?i)maintenance\b",                "Maintenance"),
    (r"(?i)escalation\b",                 "Escalation"),
    (r"(?i)policy|policies",              "Policy"),
    (r"(?i)appendi(x|ces)",               "Appendices"),
    (r"(?i)glossary\b",                   "Glossary"),
    (r"(?i)faq\b|frequently\s*asked",     "FAQ"),
]

def detect_section(text: str) -> str:
    for pattern, label in SECTION_PATTERNS:
        if re.search(pattern, text[:120]):
            return label
    return "General"


# ── PDF extraction ─────────────────────────────────────────────────────────────
def extract_pages(pdf_path: str, ocr_fn=None) -> list[dict]:
    # Scanned documents have no text layer — pdfplumber returns "" for those
    # pages. ocr_fn(pdf_path, page_num), when given, transcribes the page
    # image instead so scanned pages aren't silently dropped.
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        current_section = "General"
        for i, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            ocr = not text  # read by Gemini, so answers using it are marked
            if not text and ocr_fn is not None:
                text = ocr_fn(pdf_path, i).strip()
            if not text:
                continue
            detected = detect_section(text)
            if detected != "General":
                current_section = detected
            pages.append({"page_num": i, "text": text, "section": current_section, "ocr": ocr})
    return pages


# ── Chunking ───────────────────────────────────────────────────────────────────
def chunk_pages(pages: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", "! ", "? ", " "],
        length_function=len,
    )
    chunks, chunk_id = [], 0
    for page in pages:
        for t in splitter.split_text(page["text"]):
            t = t.strip()
            if len(t) < 40:
                continue
            chunks.append({
                "id":       f"chunk_{chunk_id:04d}",
                "text":     t,
                "page_num": page["page_num"],
                "section":  page["section"],
                "ocr":      page["ocr"],
            })
            chunk_id += 1
    return chunks


# ── File catalog + filename matching (no embeddings, no LLM) ──────────────────
# Used for "list the files" / "show me the X document" requests. A domain's
# catalog is every PDF/image under its data folder; subfolders act as projects.
FILE_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png"}

_FILE_STOPWORDS = {
    "give", "show", "please", "plan", "plans", "drawing", "drawings",
    "blueprint", "blueprints", "file", "files", "document", "documents",
    "docs", "doc", "find", "get", "want", "need", "see", "look", "related",
    "building", "permit", "permits", "order", "orders", "pdf", "pdfs", "jpg",
    "jpeg", "png", "image", "images", "picture", "pictures", "list", "tell",
    "about", "project", "projects", "available", "exist", "exists",
    "existing", "info", "information",
}

_ENGLISH_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "to", "of",
    "for", "in", "on", "at", "by", "with", "about", "as", "from", "into",
    "up", "down", "out", "off", "over", "under", "again", "further", "than",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "its",
    "they", "them", "their", "this", "that", "these", "those", "there",
    "here",
    "what", "whats", "which", "who", "whom", "where", "when", "how", "why",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "doing", "done",
    "have", "has", "had", "having",
    "can", "could", "will", "would", "shall", "should", "may", "might",
    "must",
    "not", "no", "nor", "any", "all", "each", "other", "another", "some",
    "such", "only", "own", "same", "too", "also", "very", "just", "more",
    "most", "much",
}

STOPWORDS = _FILE_STOPWORDS | _ENGLISH_STOPWORDS


def normalize_for_match(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _token_hits(token: str, normalized_spaced: str) -> bool:
    """Substring match on the space-collapsed name, or prefix match on a whole word
    (handles simple plurals/typos without a fuzzy-matching dependency)."""
    if token in normalized_spaced.replace(" ", ""):
        return True
    for word in normalized_spaced.split():
        if len(word) >= 3 and (word.startswith(token) or token.startswith(word)):
            return True
    return False


def tokenize_query(query: str) -> list[str]:
    return [t for t in normalize_for_match(query).split() if t not in STOPWORDS and len(t) > 1]


def build_file_catalog(root_dir: str) -> list[dict]:
    root = Path(root_dir)
    catalog = []
    if not root.exists():
        return catalog
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in FILE_EXTENSIONS:
            folder = path.parent.relative_to(root).as_posix()
            if folder == ".":
                folder = ""
            catalog.append({
                "path":              path,
                "rel":               path.relative_to(root).as_posix(),
                "filename":          path.name,
                "folder":            folder,
                "normalized_folder": normalize_for_match(folder),
            })
    return catalog


def find_folder_matches(query: str, catalog: list[dict], threshold: float = 0.6) -> list[dict]:
    """Pick the best-matching folder by name and return every file in it; fall
    back to filename matching for files that sit directly in the domain folder."""
    tokens = tokenize_query(query)
    log.info("[folder-match] query=%r tokens=%s", query, tokens)
    if not tokens:
        return []

    best_folder, best_score = None, 0.0
    for folder in sorted({e["folder"] for e in catalog if e["folder"]}):
        normalized = next(e["normalized_folder"] for e in catalog if e["folder"] == folder)
        score = sum(1 for t in tokens if _token_hits(t, normalized)) / len(tokens)
        if score > best_score:
            best_folder, best_score = folder, score

    if best_folder is not None and best_score >= threshold:
        matches = [e for e in catalog if e["folder"] == best_folder]
        log.info("[folder-match] MATCHED %r (score=%.2f) -> %d file(s)", best_folder, best_score, len(matches))
        return sorted(matches, key=lambda e: e["filename"])

    scored = []
    for entry in catalog:
        if entry["folder"]:
            continue
        normalized = normalize_for_match(Path(entry["filename"]).stem)
        score = sum(1 for t in tokens if _token_hits(t, normalized)) / len(tokens)
        if score >= threshold:
            scored.append({**entry, "score": round(score, 3)})
    scored.sort(key=lambda e: (-e["score"], e["filename"]))
    return scored


def find_file_by_name(query: str, catalog: list[dict], threshold: float = 0.6) -> list[dict]:
    """Filename search across the whole catalog, regardless of folder."""
    tokens = tokenize_query(query)
    if not tokens:
        return []
    scored = []
    for entry in catalog:
        normalized = normalize_for_match(Path(entry["filename"]).stem)
        score = sum(1 for t in tokens if _token_hits(t, normalized)) / len(tokens)
        if score >= threshold:
            scored.append({**entry, "score": round(score, 3)})
    scored.sort(key=lambda e: (-e["score"], e["filename"]))
    log.info("[filename-search] query=%r tokens=%s -> %s", query, tokens, [s["filename"] for s in scored])
    return scored


# ── Main RAG class ─────────────────────────────────────────────────────────────
class IntelliVaultRAG:
    EMBED_MODEL  = "gemini-embedding-001"
    CHAT_MODEL   = importlib.import_module("2_enrich_async").GEMINI_MODEL  # gemini-2.5-flash is being retired
    TOP_K        = 5
    VECTOR_DIM   = 3072
    BATCH_SIZE   = 25
    MAX_WORKERS  = 4   # parallel embed/vision calls during indexing

    SYSTEM_PROMPT = (
        "You are TekMark's IntelliVault, an expert AI knowledge assistant with deep "
        "knowledge of operations manuals, procedures, and support documentation across "
        "multiple domains (Construction, Event Management, IT Help Desk, Manufacturing).\n\n"
        "Guidelines:\n"
        "- Answer ONLY from the provided context chunks.\n"
        "- Context chunks may be text excerpts OR vision-generated descriptions of "
        "images/diagrams (tagged as image_caption). Treat both equally.\n"
        "- Be precise, professional and helpful.\n"
        "- If the context does not cover the question, say so clearly.\n"
        "- Use bullet points where appropriate for clarity.\n"
        "- Keep answers concise but complete.\n"
        "- Do NOT mention page numbers, section names, or source references in your answer."
    )

    def __init__(self, gemini_api_key: str, chroma_path: str = "chroma_db"):
        self.client = genai.Client(
            api_key=gemini_api_key,
            http_options=types.HttpOptions(timeout=120_000),  # 120s, large PDFs can be slow
        )
        self.db = chromadb.PersistentClient(path=chroma_path)

    def _col(self, namespace: str):
        """One cosine collection per domain slug (Chroma names need >= 3 chars, hence the prefix)."""
        return self.db.get_or_create_collection(f"iv_{namespace or 'default'}", metadata={"hnsw:space": "cosine"})

    def _upsert(self, namespace: str, vectors: list[dict]) -> None:
        self._col(namespace).upsert(
            ids=[v["id"] for v in vectors],
            embeddings=[v["values"] for v in vectors],
            metadatas=[v["metadata"] for v in vectors],
        )

    def _search(self, namespace: str, vec, top_k: int, types_: list[str]) -> list[tuple[dict, float]]:
        """Returns (metadata, cosine similarity) pairs, best first."""
        col = self._col(namespace)
        if col.count() == 0:
            return []
        r = col.query(query_embeddings=[vec], n_results=top_k, where={"type": {"$in": types_}})
        return [(m, 1 - d) for m, d in zip(r["metadatas"][0], r["distances"][0])]

    def _call_with_retry(self, fn, *args, max_retries: int = 4, **kwargs):
        delay = 2.0
        for attempt in range(max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                msg = str(e)
                if (
                    "503" in msg or "UNAVAILABLE" in msg
                    or "429" in msg or "RESOURCE_EXHAUSTED" in msg
                    or "timed out" in msg.lower() or "timeout" in msg.lower()
                ):
                    if attempt == max_retries - 1:
                        raise
                    wait = delay * (2 ** attempt) + random.uniform(0, 1)
                    log.warning("Gemini API unavailable (attempt %d/%d), retrying in %.1fs…", attempt + 1, max_retries, wait)
                    time.sleep(wait)
                else:
                    raise

    def _embed_batch(self, texts: list[str], task: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
        response = self._call_with_retry(
            self.client.models.embed_content,
            model=self.EMBED_MODEL,
            contents=texts,
            config=types.EmbedContentConfig(task_type=task),
        )
        return [e.values for e in response.embeddings]

    def _embed_query(self, query: str) -> list[float]:
        return self._embed_batch([query], task="RETRIEVAL_QUERY")[0]

    def _is_visual_pdf(self, pdf_path: str, sample_pages: int = 3) -> bool:
        """Returns True if the PDF has very little extractable text — treat as image-heavy."""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages_to_check = list(pdf.pages[:sample_pages])
                total_chars = sum(len((p.extract_text() or "")) for p in pages_to_check)
                avg_chars = total_chars / max(len(pages_to_check), 1)
            return avg_chars < 80
        except Exception:
            return False

    def _caption_page(self, pdf_path: str, page_num: int) -> str:
        """Renders a PDF page and asks Gemini Vision to describe its visual content."""
        doc = fitz.open(pdf_path)
        try:
            png_bytes = doc[page_num - 1].get_pixmap(dpi=200).tobytes("png")
        finally:
            doc.close()
        return self._caption_image_bytes(
            png_bytes,
            "You are analyzing a page from an operations/support document that is "
            "mostly image-based (diagram, screenshot, chart, or scanned figure). "
            "Describe in detail what is shown: the type of image, all visible labels, "
            "steps, callouts, and any notable features. Be specific and thorough — "
            "this description will be used to answer questions about it. "
            "Output only the description, no commentary.",
        )

    def _caption_image_file(self, image_path: str) -> str:
        """Reads a standalone image file and asks Gemini Vision to describe its content."""
        mime = "image/jpeg" if Path(image_path).suffix.lower() in (".jpg", ".jpeg") else "image/png"
        img_bytes = Path(image_path).read_bytes()
        return self._caption_image_bytes(
            img_bytes,
            "You are analyzing an image from a company knowledge base (could be a "
            "screenshot, photo of equipment, diagram, or scanned document). "
            "Describe in detail what is shown: the type of image, all visible text, "
            "labels, steps, and any notable features. Be specific and thorough — "
            "this description will be used to answer support questions about it. "
            "Output only the description, no commentary.",
            mime=mime,
        )

    def _caption_image_bytes(self, img_bytes: bytes, instruction: str, mime: str = "image/png") -> str:
        try:
            response = self._call_with_retry(
                self.client.models.generate_content,
                model=self.CHAT_MODEL,
                contents=[
                    types.Part.from_bytes(data=img_bytes, mime_type=mime),
                    instruction,
                ],
                config=types.GenerateContentConfig(temperature=0),
            )
            return response.text or ""
        except Exception as e:
            log.warning("captioning failed: %s", e)
            return ""

    def _ocr_page(self, pdf_path: str, page_num: int) -> str:
        # Renders a page (1-indexed, matching pdfplumber) and asks Gemini to
        # transcribe it verbatim — covers scanned pages with no text layer.
        doc = fitz.open(pdf_path)
        try:
            png_bytes = doc[page_num - 1].get_pixmap(dpi=200).tobytes("png")
        finally:
            doc.close()

        try:
            response = self._call_with_retry(
                self.client.models.generate_content,
                model=self.CHAT_MODEL,
                contents=[
                    types.Part.from_bytes(data=png_bytes, mime_type="image/png"),
                    "Transcribe ALL visible text on this scanned document page "
                    "verbatim, preserving labels and values (e.g. 'Field: value'). "
                    "Include stamps, headers, and handwritten text if legible. "
                    "Output only the transcribed text, no commentary.",
                ],
                config=types.GenerateContentConfig(temperature=0),
            )
            return response.text or ""
        except Exception as e:
            log.warning("[%s] OCR failed on page %d: %s", Path(pdf_path).name, page_num, e)
            return ""

    # ── Intent routing + file lookup ─────────────────────────────────────────
    # Stage A: one forced-function-call Gemini request classifies the query into
    # exactly one of three tools. Stage B: plain Python resolves the extracted
    # strings against the domain's file catalog — no second LLM call, except an
    # optional cheap narrowing call when several files in a folder match.
    INTENT_TOOLS = [
        {
            "name": "get_list_of_files",
            "description": (
                "Use when the user wants a listing — either the files in one specific "
                "project/folder, or an overview of everything available (e.g. 'what "
                "documents do we have', 'list all files', 'what's in the Westside "
                "project'). Leave project empty for 'everything'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "The project/folder name as it appears in the query, verbatim. Empty if the user means everything.",
                    },
                },
            },
        },
        {
            "name": "get_specific_file",
            "description": (
                "Use when the user wants to see or get a specific document, drawing, "
                "or file (e.g. 'show me the operations manual', 'give me the floor "
                "plan'), as opposed to a listing of everything."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "The project/folder name as it appears in the query, if any. Empty if not mentioned.",
                    },
                    "file_hint": {
                        "type": "string",
                        "description": "The document type or filename fragment the user is asking for, verbatim from the query.",
                    },
                },
                "required": ["file_hint"],
            },
        },
        {
            "name": "answer_from_context",
            "description": (
                "Use for any substantive question that needs an answer generated from "
                "document content — procedures, policies, specifications, troubleshooting, "
                "anything requiring retrieval + reasoning over the indexed documents. "
                "This is the default when the query isn't clearly a file listing or a "
                "specific-file request."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The user's question, verbatim."},
                },
                "required": ["query"],
            },
        },
    ]

    def detect_intent(self, query: str, chat_history: list[dict] | None = None, domain_label: str = "") -> dict:
        """Single forced function-call to Gemini. Falls back to answer_from_context
        on any failure so routing never blocks the main RAG path."""
        history_block = ""
        if chat_history:
            for msg in chat_history[-6:]:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_block += f"{role}: {msg['content']}\n"

        prompt = (
            f"You are the intent router for a {domain_label or 'company'} document assistant. "
            "Classify the user's request into exactly one tool call.\n\n"
        )
        if history_block:
            prompt += f"CONVERSATION HISTORY:\n{history_block}\n"
        prompt += f"USER REQUEST: {query}"

        try:
            response = self._call_with_retry(
                self.client.models.generate_content,
                model=self.CHAT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0,
                    tools=[types.Tool(function_declarations=self.INTENT_TOOLS)],
                    tool_config=types.ToolConfig(
                        function_calling_config=types.FunctionCallingConfig(mode="ANY")
                    ),
                ),
            )
            call = response.candidates[0].content.parts[0].function_call
            args = dict(call.args) if call.args else {}
            log.info("[intent] query=%r -> %s(%s)", query, call.name, args)
            return {"name": call.name, "args": args}
        except Exception as e:
            log.warning("[intent] detection failed (%s), defaulting to answer_from_context", e)
            return {"name": "answer_from_context", "args": {"query": query}}

    def _narrow_to_specific_files(self, query: str, entries: list[dict]) -> list[dict]:
        """One cheap Gemini text call to pick the specific file(s) the query wants
        out of a folder; falls back to all of them."""
        filenames = [e["filename"] for e in entries]
        prompt = (
            "You are matching a user's request to files in a folder.\n\n"
            "Files available:\n" + "\n".join(f"- {f}" for f in filenames) +
            f"\n\nUser request: \"{query}\"\n\n"
            "If the request clearly points to one or more SPECIFIC files from "
            "the list (e.g. it names a document type, a version, or wording close "
            "to a filename), reply with ONLY those exact filenames from the list, "
            "one per line, nothing else.\n"
            "If the request is general (just asking for 'files', 'everything', "
            "etc.) and doesn't point to anything specific, reply with exactly: ALL"
        )
        try:
            response = self._call_with_retry(
                self.client.models.generate_content,
                model=self.CHAT_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0),
            )
            text = response.text.strip()
        except Exception as e:
            log.warning("[narrow] Gemini call failed, returning all candidates: %s", e)
            return entries

        if text.upper() == "ALL":
            return entries
        selected = {line.strip("-* ").strip() for line in text.splitlines() if line.strip()}
        matched = [e for e in entries if e["filename"] in selected]
        return matched if matched else entries

    @staticmethod
    def _resolve_project(project_hint: str, catalog: list[dict]) -> str | None:
        if not project_hint:
            return None
        matches = find_folder_matches(project_hint, catalog)
        return matches[0]["folder"] if matches else None

    def get_list_of_files(self, catalog: list[dict], project: str | None = None, root_label: str = "General") -> dict:
        """{'type': 'file_list', 'projects': {folder: [filenames]}}. Empty project -> all.
        Files sitting directly in the domain folder are grouped under root_label (the project name)."""
        if not project:
            by_folder: dict[str, list[str]] = {}
            for e in sorted(catalog, key=lambda x: (x["folder"], x["filename"])):
                by_folder.setdefault(e["folder"] or root_label, []).append(e["filename"])
            return {"type": "file_list", "projects": by_folder}

        resolved = self._resolve_project(project, catalog)
        if not resolved:
            return {"type": "file_list", "projects": {}, "query_project": project}
        files = sorted(e["filename"] for e in catalog if e["folder"] == resolved)
        return {"type": "file_list", "projects": {resolved: files}}

    def get_specific_file(self, catalog: list[dict], project: str | None, file_hint: str) -> dict:
        """{'type': 'specific_file', 'matches': [catalog entries]}."""
        resolved = self._resolve_project(project, catalog) if project else None
        if resolved:
            candidates = [e for e in catalog if e["folder"] == resolved]
        else:
            candidates = find_folder_matches(file_hint or "", catalog)
            if not candidates:
                candidates = find_file_by_name(file_hint or "", catalog)
        if not candidates and file_hint:  # e.g. "plan 1": only stopwords, so let Gemini pick from all files
            picked = self._narrow_to_specific_files(file_hint, catalog)
            return {"type": "specific_file", "matches": picked if len(picked) < len(catalog) else []}
        matches = self._narrow_to_specific_files(file_hint or "", candidates) if len(candidates) > 1 else candidates
        return {"type": "specific_file", "matches": matches}

    def route(self, query: str, catalog: list[dict], chat_history: list[dict] | None = None,
              namespace: str = "", domain_label: str = "") -> dict:
        """Main entry point: one intent-detection call, then dispatch to the matching tool."""
        intent = self.detect_intent(query, chat_history, domain_label)
        name, args = intent["name"], intent.get("args", {})
        # Gemini sometimes fills an empty optional string with a placeholder like "None".
        project = args.get("project")
        if not project or str(project).strip().lower() in {"none", "null", "n/a", "all", "everything", "any"}:
            project = None

        if name == "get_list_of_files":
            return self.get_list_of_files(catalog, project, root_label=domain_label or "General")
        if name == "get_specific_file":
            return self.get_specific_file(catalog, project, args.get("file_hint", query))

        result = self.query(query, chat_history=chat_history, namespace=namespace, domain_label=domain_label)
        result["type"] = "answer"
        return result

    # ── Indexing ─────────────────────────────────────────────────────────────
    def index_document(self, pdf_path: str, source_label: str | None = None, category: str = "", namespace: str = "") -> int:
        source = source_label or Path(pdf_path).name
        vector_id_prefix = re.sub(r"[^A-Za-z0-9_.-]", "_", source)
        t0 = time.time()
        log.info("[%s] extracting text...", source)

        pages = extract_pages(pdf_path, ocr_fn=self._ocr_page)
        log.info("[%s] %d pages with text extracted", source, len(pages))

        chunks = chunk_pages(pages)
        log.info("[%s] split into %d chunks", source, len(chunks))
        if not chunks:
            return 0

        batches = [chunks[i: i + self.BATCH_SIZE] for i in range(0, len(chunks), self.BATCH_SIZE)]
        log.info("[%s] embedding %d batches (batch_size=%d, workers=%d)",
                  source, len(batches), self.BATCH_SIZE, self.MAX_WORKERS)

        def embed_one(idx: int, batch: list[dict]) -> tuple[int, list[dict], list[list[float]]]:
            bt0 = time.time()
            embeddings = self._embed_batch([c["text"] for c in batch])
            log.info("[%s] batch %d/%d embedded (%d chunks) in %.1fs",
                      source, idx + 1, len(batches), len(batch), time.time() - bt0)
            return idx, batch, embeddings

        with ThreadPoolExecutor(max_workers=self.MAX_WORKERS) as executor:
            futures = [executor.submit(embed_one, i, b) for i, b in enumerate(batches)]
            for future in as_completed(futures):
                idx, batch, embeddings = future.result()
                self._upsert(namespace, [
                    {
                        "id":     f"{vector_id_prefix}_{c['id']}",
                        "values": emb,
                        "metadata": {
                            "type":     "text",
                            "text":     c["text"],
                            "page":     c["page_num"],
                            "section":  c["section"],
                            "ocr":      c["ocr"],
                            "source":   source,
                            "category": category,
                        },
                    }
                    for c, emb in zip(batch, embeddings)
                ])
                log.info("[%s] batch %d/%d upserted to ChromaDB", source, idx + 1, len(batches))

        log.info("[%s] done: %d chunks indexed in %.1fs", source, len(chunks), time.time() - t0)
        return len(chunks)

    def index_visual_document(self, pdf_path: str, source_label: str | None = None, category: str = "", namespace: str = "") -> int:
        """Captions each page of an image-heavy PDF using Gemini Vision and indexes the captions."""
        source = source_label or Path(pdf_path).name
        vector_id_prefix = re.sub(r"[^A-Za-z0-9_.-]", "_", source)
        t0 = time.time()

        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
        except Exception as e:
            log.warning("[%s] could not open for captioning: %s", source, e)
            return 0

        log.info("[%s] captioning %d pages (visual document)...", source, page_count)
        captions = []
        for page_num in range(1, page_count + 1):
            caption = self._caption_page(pdf_path, page_num)
            if caption.strip():
                captions.append({"page_num": page_num, "caption": caption})
                log.info("[%s] page %d captioned (%d chars)", source, page_num, len(caption))

        if not captions:
            log.warning("[%s] no captions generated", source)
            return 0

        texts = [c["caption"] for c in captions]
        embeddings = self._embed_batch(texts)
        self._upsert(namespace, [
            {
                "id":     f"{vector_id_prefix}_cap_p{c['page_num']}",
                "values": emb,
                "metadata": {
                    "type":     "image_caption",
                    "text":     c["caption"],
                    "page":     c["page_num"],
                    "section":  "Visual Content",
                    "source":   source,
                    "category": category,
                },
            }
            for c, emb in zip(captions, embeddings)
        ])
        log.info("[%s] done: %d page captions indexed in %.1fs", source, len(captions), time.time() - t0)
        return len(captions)

    def index_image(self, image_path: str, source_label: str | None = None, category: str = "", namespace: str = "") -> int:
        """Captions a standalone image file using Gemini Vision and indexes the caption."""
        source = source_label or Path(image_path).name
        vector_id_prefix = re.sub(r"[^A-Za-z0-9_.-]", "_", source)

        caption = self._caption_image_file(image_path)
        if not caption.strip():
            log.warning("[%s] no caption generated", source)
            return 0

        embedding = self._embed_batch([caption])[0]
        self._upsert(namespace, [{
            "id":     f"{vector_id_prefix}_img",
            "values": embedding,
            "metadata": {
                "type":     "image_caption",
                "text":     caption,
                "page":     1,
                "section":  "Visual Content",
                "source":   source,
                "category": category,
            },
        }])
        log.info("[%s] image caption indexed", source)
        return 1

    def _is_indexed(self, source: str, namespace: str) -> bool:
        """True if any vector for this file (id prefix = sanitized source) exists."""
        prefix = re.sub(r"[^A-Za-z0-9_.-]", "_", source) + "_"
        return bool(self._col(namespace).get(where={"source": source}, limit=1)["ids"])

    def index_folder(self, data_dir: str, namespace: str = "", only_new: bool = False) -> dict[str, int]:
        """Recursively indexes every PDF and image under data_dir. The top-level
        subfolder name (e.g. 'Construction', 'IT Help Desk') is stored as the
        'category' metadata field for each chunk/caption. With only_new=True,
        files that already have vectors in the namespace are skipped."""
        data_path = Path(data_dir)
        t0 = time.time()
        text_count, caption_count = 0, 0

        if not data_path.exists():
            return {"text_chunks": 0, "image_captions": 0}

        for file_path in sorted(data_path.rglob("*")):
            if not file_path.is_file():
                continue

            rel = file_path.relative_to(data_path).as_posix()
            parts = file_path.relative_to(data_path).parts
            category = parts[0] if len(parts) > 1 else namespace

            suffix = file_path.suffix.lower()
            if suffix != ".pdf" and suffix not in IMAGE_EXTENSIONS:
                continue
            if only_new and self._is_indexed(rel, namespace):
                continue
            log.info("[%s] new file, indexing into namespace %r", rel, namespace)

            if suffix == ".pdf":
                is_visual = self._is_visual_pdf(str(file_path))
                if is_visual:
                    log.info("[%s] detected as image-heavy — using vision captioning", rel)
                    caption_count += self.index_visual_document(str(file_path), source_label=rel, category=category, namespace=namespace)
                else:
                    text_count += self.index_document(str(file_path), source_label=rel, category=category, namespace=namespace)
            elif suffix in IMAGE_EXTENSIONS:
                caption_count += self.index_image(str(file_path), source_label=rel, category=category, namespace=namespace)

        log.info("Finished indexing folder %s: %d text chunks, %d image captions in %.1fs",
                  data_dir, text_count, caption_count, time.time() - t0)
        return {"text_chunks": text_count, "image_captions": caption_count}

    # ── Direct image fetch (no LLM answer call) ─────────────────────────────
    def find_image(self, query: str, namespace: str = "", min_score: float = 0.72) -> dict | None:
        """Embedding search restricted to image captions, no generate_content call.
        Any query whose best match is an indexed diagram/image returns that
        image directly instead of going through the full RAG+LLM answer path."""
        q_vec = self._embed_query(query)
        results = self._search(namespace, q_vec, 1, ["image_caption"])
        if not results:
            return None
        meta, score = results[0]
        if score < min_score:
            return None
        return {
            "source":   meta.get("source", "?"),
            "category": meta.get("category", ""),
            "score":    round(score, 3),
        }

    # ── Query ────────────────────────────────────────────────────────────────
    def query(self, question: str, chat_history: list[dict] | None = None,
              namespace: str = "", domain_label: str = "") -> dict[str, Any]:
        q_vec = self._embed_query(question)
        results = self._search(namespace, q_vec, self.TOP_K, ["text", "image_caption"])

        sources, context_parts = [], []
        for meta, score in results:
            text = meta.get("text", "")
            sources.append({
                "text":     text,
                "page":     meta.get("page", "?"),
                "section":  meta.get("section", "General"),
                "source":   meta.get("source", "?"),
                "category": meta.get("category", ""),
                "type":     meta.get("type", "text"),
                "ocr":      meta.get("ocr", False),
                "score":    round(score, 3),
            })
            context_parts.append(f"[Source: {meta.get('source', '?')}]\n{text}")

        context = "\n\n---\n\n".join(context_parts)

        # Build chat history block
        history_block = ""
        if chat_history:
            recent = chat_history[-6:]  # last 3 user-assistant pairs
            for msg in recent:
                role = "User" if msg["role"] == "user" else "Assistant"
                history_block += f"{role}: {msg['content']}\n"

        prompt = self.SYSTEM_PROMPT
        if domain_label:
            prompt += f"\n\nThe user is currently asking about the {domain_label} domain."
        prompt += f"\n\nCONTEXT:\n{context}"
        if history_block:
            prompt += f"\n\nCONVERSATION HISTORY:\n{history_block}"
        prompt += f"\n\nQUESTION: {question}\n\nANSWER:"

        # ── Structured query log ───────────────────────────────────────────────
        sep = "=" * 60
        lines = [
            sep,
            f"DOMAIN: {domain_label or namespace or '-'}",
            f"USER PROMPT: {question}",
            "",
            f"RETRIEVED CHUNKS ({len(sources)}):",
        ]
        for i, s in enumerate(sources, 1):
            lines += [
                f"  [{i}] score={s['score']:.3f} | category={s['category']} | section={s['section']} | source={s['source']}",
                f"      {s['text'].replace(chr(10), ' ')}",
            ]
        lines.append(sep)
        _query_log.info("\n".join(lines))
        # ──────────────────────────────────────────────────────────────────────

        response = self._call_with_retry(
            self.client.models.generate_content, model=self.CHAT_MODEL, contents=prompt
        )
        return {"answer": response.text.strip(), "sources": sources}
