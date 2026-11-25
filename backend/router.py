"""Intent router (adapted from IntelliVault rag_pipeline.py): ONE forced-function-call Gemini request
classifies the question into a tool, then plain Python resolves it against the project's sheets/files.

Tools: show_sheet | show_differences | list_contents | show_file | small_talk | answer_from_context (default)
       + answer_from_documents when the project has other documents.
"""
from __future__ import annotations

import importlib
import logging
import re
from pathlib import Path

from google import genai
from google.genai import types

enrich = importlib.import_module("2_enrich_async")  # the model and its thinking setting are defined there
log = logging.getLogger("blueprint")
MAX_SHEET_CARDS = 8

# ── Filename matching helpers: copied as-is from IntelliVault rag_pipeline.py ──────────────────
_FILE_STOPWORDS = {
    "give", "show", "please", "plan", "plans", "drawing", "drawings",
    "blueprint", "blueprints", "file", "files", "document", "documents",
    "docs", "doc", "find", "get", "want", "need", "see", "look", "related",
    "building", "permit", "permits", "order", "orders", "pdf", "pdfs", "jpg",
    "jpeg", "png", "image", "images", "picture", "pictures", "list", "tell",
    "about", "project", "projects", "available", "exist", "exists",
    "existing", "info", "information", "set", "sets", "sheet", "sheets", "me",
}
_ENGLISH_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "to", "of",
    "for", "in", "on", "at", "by", "with", "about", "as", "from", "into",
    "up", "down", "out", "off", "over", "under", "again", "further", "than",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "its",
    "they", "them", "their", "this", "that", "these", "those", "there", "here",
    "what", "whats", "which", "who", "whom", "where", "when", "how", "why",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "doing", "done", "have", "has", "had", "having",
    "can", "could", "will", "would", "shall", "should", "may", "might", "must",
    "not", "no", "nor", "any", "all", "each", "other", "another", "some",
    "such", "only", "own", "same", "too", "also", "very", "just", "more", "most", "much",
}
STOPWORDS = _FILE_STOPWORDS | _ENGLISH_STOPWORDS


def normalize_for_match(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def _token_hits(token: str, normalized_spaced: str) -> bool:
    """Substring match on the space-collapsed name, or prefix match on a whole word."""
    if token in normalized_spaced.replace(" ", ""):
        return True
    for word in normalized_spaced.split():
        if len(word) >= 3 and (word.startswith(token) or token.startswith(word)):
            return True
    return False


def tokenize_query(query: str) -> list[str]:
    return [t for t in normalize_for_match(query).split() if t not in STOPWORDS and len(t) > 1]


def match_names(query: str, names: list[str], threshold: float = 0.6) -> list[str]:
    """Names whose normalized stem contains enough of the query tokens (IV's find_file_by_name)."""
    tokens = tokenize_query(query)
    if not tokens:
        return []
    scored = []
    for name in names:
        normalized = normalize_for_match(Path(name).stem)
        score = sum(1 for t in tokens if _token_hits(t, normalized)) / len(tokens)
        if score >= threshold:
            scored.append((score, name))
    return [n for _, n in sorted(scored, key=lambda x: (-x[0], x[1]))]


# ── Stage A: intent detection ─────────────────────────────────────────────────────────────────
INTENT_TOOLS = [
    {
        "name": "show_sheet",
        "description": (
            "Use when the user wants to SEE one or more drawing sheets, by sheet number "
            "(e.g. 'show A101', 'pull up S-201 and S202') or by sheet name (e.g. 'show me the "
            "second floor plan', 'open the roof framing plan'). NOT for differences or changes between versions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sheet_ids": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Sheet numbers exactly as written in the request, e.g. ['A101']. Empty if none given.",
                },
                "description": {
                    "type": "string",
                    "description": "The sheet name/description from the request if no number is given, else empty.",
                },
            },
        },
    },
    {
        "name": "show_differences",
        "description": (
            "Use when the user asks what is different, changed, revised, added or deleted between the "
            "versions v1 and v2: the whole set ('difference between v1 and v2', 'what changed') or "
            "specific sheets ('show me the differences in A101', 'what changed on the foundation plan')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sheet_ids": {
                    "type": "array", "items": {"type": "string"},
                    "description": "Sheet numbers exactly as written, e.g. ['A101']. Empty for the whole set.",
                },
                "description": {
                    "type": "string",
                    "description": "Sheet NAME if one is given instead of a number (e.g. 'foundation plan'). "
                                   "Empty for the whole set; never 'v1', 'v2' or 'all'.",
                },
            },
        },
    },
    {
        "name": "list_contents",
        "description": (
            "Use when the user wants a listing: which files/sets are in the project, or the sheet "
            "index of a set (e.g. 'what sheets are there', 'list the electrical sheets in v2', "
            "'what documents do we have')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file_hint": {"type": "string", "description": "File/set name if one is mentioned, else empty."},
                "discipline": {
                    "type": "string",
                    "description": "Discipline filter if mentioned (Architectural, Structural, Electrical, "
                                   "Mechanical, Plumbing, Civil, Life Safety, Fire Protection), else empty.",
                },
            },
        },
    },
    {
        "name": "show_file",
        "description": (
            "Use when the user wants a whole PDF file / drawing set, not a single sheet "
            "(e.g. 'give me the v2 set', 'download the full construction set pdf')."
        ),
        "parameters": {
            "type": "object",
            "properties": {"file_hint": {"type": "string", "description": "File/set name fragment from the request."}},
            "required": ["file_hint"],
        },
    },
    {
        "name": "small_talk",
        "description": (
            "Use ONLY for greetings, thanks, goodbyes, or questions about the assistant itself "
            "(e.g. 'hi', 'thanks!', 'what can you do?'). Never for questions about the drawings."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reply": {
                    "type": "string",
                    "description": "A short friendly reply (1-2 sentences). If asked what you can do: you can "
                                   "show sheets by number or name (e.g. 'show A101'), list the sheets/files in "
                                   "the project, open a whole PDF set, and answer questions about the drawings.",
                },
            },
            "required": ["reply"],
        },
    },
    {
        "name": "answer_from_context",
        "description": (
            "Use for any question that needs an answer from the drawings' content: dimensions, "
            "notes, schedules, materials, code requirements, where something is. "
            "This is the default when the request isn't clearly a sheet view, listing, or file request."
        ),
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The user's question, verbatim."}},
            "required": ["query"],
        },
    },
]

# Offered only in projects that have other documents.
DOCUMENTS_TOOL = {
    "name": "answer_from_documents",
    "description": (
        "Use for any question whose answer is in the project's OTHER documents (contracts, quotes, manuals, "
        "letters, photos), not in the drawings: prices, parties, dates, payment and insurance terms, scope of "
        "a contract, warranties, or when a listed document is named. Use answer_from_context for what is on "
        "the drawings: rooms, dimensions, schedules, materials, building code data."
    ),
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "The user's question, verbatim."}},
        "required": ["query"],
    },
}


def detect_intent(client: genai.Client, query: str, history: list[dict] | None = None, project_label: str = "",
                  drawings: list[str] = (), documents: list[str] = ()) -> dict:
    """Single forced function call. Falls back to answer_from_context on any failure so routing
    never blocks the main RAG path. `drawings` / `documents`: the project's file names."""
    prompt = (
        f"You are the intent router for an architectural blueprint assistant (project: {project_label or '-'}). "
        "Classify the user's request into exactly one tool call.\n\n"
    )
    if documents:
        prompt += (f"DRAWING SETS: {', '.join(drawings) or '(none)'}\n"
                   f"OTHER DOCUMENTS: {', '.join(documents[:100])}\n\n")
    if history:
        lines = [f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content'][:500]}" for m in history[-6:]]
        prompt += "CONVERSATION HISTORY:\n" + "\n".join(lines) + "\n\n"
    prompt += f"USER REQUEST: {query}"
    try:
        response = client.models.generate_content(
            model=enrich.GEMINI_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0,
                tools=[types.Tool(function_declarations=INTENT_TOOLS + ([DOCUMENTS_TOOL] if documents else []))],
                tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY")),
                thinking_config=enrich.NO_THINKING,
            ),
        )
        call = response.candidates[0].content.parts[0].function_call
        return {"name": call.name, "args": dict(call.args or {})}
    except Exception as e:
        log.warning(f"[router] intent detection failed ({type(e).__name__}: {e}), using answer_from_context")
        return {"name": "answer_from_context", "args": {"query": query}}


# ── Stage B: resolvers (plain Python over the project's indexed sheets) ────────────────────────
def norm_sheet_id(s: str) -> str:
    """'a-101' / 'A 101' / 'A101' -> 'A101'; keeps decimals like M1.01."""
    return re.sub(r"[^A-Z0-9.]", "", s.upper())


def find_sheets(corpus: list[dict], sheet_ids: list[str], description: str) -> list[dict]:
    """Every corpus item (all files) matching the IDs; else title token-match on the description."""
    wanted = {norm_sheet_id(s) for s in sheet_ids if norm_sheet_id(s)}
    if wanted:
        hits = [c for c in corpus if norm_sheet_id(c["metadata"]["sheet_id"]) in wanted]
        if hits:
            return _sorted_sheets(hits)
    text = description or " ".join(sheet_ids)
    tokens = tokenize_query(text)
    if not tokens:
        return []
    scored = []
    for c in corpus:
        title = normalize_for_match(c["metadata"]["sheet_title"])
        score = sum(1 for t in tokens if _token_hits(t, title)) / len(tokens)
        if score >= 0.75:
            scored.append((score, c))
    best = max((s for s, _ in scored), default=0)
    return _sorted_sheets([c for s, c in scored if s == best])


def _sorted_sheets(items: list[dict]) -> list[dict]:
    return sorted(items, key=lambda c: (c["metadata"].get("file", ""), c["metadata"]["page_number"]))


def list_contents(corpus: list[dict], file_hint: str = "", discipline: str = "") -> str:
    by_file: dict[str, list[dict]] = {}
    for c in corpus:
        by_file.setdefault(c["metadata"].get("file") or c["metadata"]["version"], []).append(c["metadata"])
    files = sorted(by_file)
    if file_hint:
        files = match_names(file_hint, files) or files
    lines = []
    for f in files:
        sheets = sorted(by_file[f], key=lambda m: m["page_number"])
        if discipline:
            sheets = [m for m in sheets if discipline.lower() in str(m.get("discipline", "")).lower()]
        if not sheets:
            continue
        lines.append(f"**{f}** ({len(sheets)} sheets)\n")
        lines += [f"- **{m['sheet_id']}** – {m['sheet_title']}" for m in sheets]
        lines.append("")
    return "\n".join(lines) if lines else "No matching sheets found."
