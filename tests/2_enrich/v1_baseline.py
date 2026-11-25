import asyncio
import io
import json
import logging
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from dotenv import load_dotenv

load_dotenv()
from google.genai import errors, types

log = logging.getLogger("blueprint")


class SheetSpatialMetadata(BaseModel):
    verified_sheet_id: str = Field(
        description="The exact Sheet ID printed in the bottom-right title block box (e.g., 'A001', 'S202', 'M101')."
    )
    verified_sheet_title: str = Field(
        description="The exact Sheet Title printed in the title block (e.g., 'SECOND FLOOR FRAMING PLAN')."
    )
    discipline: str = Field(
        description="Primary discipline: 'Architectural', 'Life Safety', 'Civil', 'Structural', 'Mechanical', 'Electrical', 'Plumbing', or 'Fire Protection'."
    )
    floor_level: str = Field(
        description="Applicable floor or scope: 'General/Cover', 'Site', 'Foundation', '1st Floor', '2nd Floor', '3rd Floor', '4th Floor', 'Roof', or 'Details/Sections'."
    )
    grid_bounds: str = Field(
        description="Structural grid range visible on the drawing (e.g., 'Horizontal 1-25, Vertical A-S'), or 'N/A' for schedules/notes/details."
    )
    referenced_sheets: list[str] = Field(
        default_factory=list,
        description="Other sheet IDs explicitly referenced in callouts or notes on this page (e.g., ['S102', 'S103', 'S201'])."
    )
    key_entities: list[str] = Field(
        default_factory=list,
        description="Important room names/numbers, structural members (e.g., 'W24x55', 'F-9', 'C-1'), equipment, or code sections shown."
    )
    spatial_summary: str = Field(
        description="Detailed 4-6 sentence description of the layout, spatial relationships between wings/rooms/grids, schedules, and key construction notes."
    )


GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
# ponytail: calibration knob. Smaller = faster upload, but tiny title-block text gets harder to read.
VLM_MAX_SIDE = int(os.getenv("VLM_MAX_SIDE", "3072"))
MAX_OUTPUT_TOKENS = int(os.getenv("ENRICH_MAX_OUTPUT_TOKENS", "4096"))
# Per-request timeout; a hung call otherwise blocks its file forever (the SDK has none by default).
HTTP_OPTIONS = {"api_version": "v1beta", "timeout": int(os.getenv("GEMINI_TIMEOUT_S", "120")) * 1000}


SHEET_ID_RE = re.compile(r"[A-Z]{1,3}[-.]?\d{1,4}(?:\.\d+)?[A-Z]?")  # A101, GR-325, M1.01, A402A


def clean_sheet_id(raw: str) -> str:
    """Normalized sheet number, or "" if `raw` doesn't look like one (e.g. '1/8', 'N/A', 'Page 2')."""
    s = re.sub(r"\s+", "", raw or "").upper()
    return s if SHEET_ID_RE.fullmatch(s) else ""


def vlm_image_part(image_path: str) -> types.Part:
    """Downscaled JPEG for the VLM only; the full-res PNG on disk is kept for chat answers.
    Cuts upload from ~2 MB PNG to ~0.4 MB per page."""
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img.thumbnail((VLM_MAX_SIDE, VLM_MAX_SIDE))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
    return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")


async def enrich_single_sheet(
    client: genai.Client,
    sheet: dict,
    semaphore: asyncio.Semaphore,
    max_retries: int = 4,
    tag: str = "",
) -> dict:
    tag = tag or sheet.get("version", "")
    page = f"page {sheet['page_number']} ({sheet['sheet_id']})"
    async with semaphore:
        prompt = f"""
You are a senior architectural & structural engineer indexing a construction blueprint sheet for a RAG system.
Analyze both the attached blueprint image and its extracted CAD text below.

Parsed Sheet ID (may be UNKNOWN if CAD font was vectorized—always verify against the bottom-right title block box in the image): {sheet['sheet_id']}
Parsed Sheet Title: {sheet['sheet_title']}
Revision Date: {sheet['revision_date']}

Extracted CAD Text (truncated if large):
{sheet['raw_cad_text'][:12000]}

Extract accurate structured metadata:
1. Read `verified_sheet_id` and `verified_sheet_title` directly from the title block in the image.
2. Describe the spatial layout (e.g., Hampton Inn wing on the right/east vs Home2 Suites wing on the left/west, pool area, elevator core, grid intersections).
3. Extract cross-sheet detail references (e.g., 'SEE 2/S103' -> 'S103', 'SEE S201' -> 'S201').
4. List key schedules, footing types, beam sizes, room numbers, or occupant load tables present on the sheet.
"""
        image_part = await asyncio.to_thread(vlm_image_part, sheet["image_path"])
        for attempt in range(1, max_retries + 1):
            try:
                response = await client.aio.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[image_part, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=SheetSpatialMetadata,
                        temperature=0.1,
                        # Normal output is ~500 tokens. Text-heavy notes sheets can make the model loop on
                        # key_entities until the 65k default cap (minutes per page); cut that off early.
                        max_output_tokens=MAX_OUTPUT_TOKENS,
                        # Structured extraction doesn't need reasoning; thinking tokens are billed as output.
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )

                meta = json.loads(response.text)

                # Backfill Sheet ID & Title if Step 1 had marked them UNKNOWN / UNTITLED
                updated = {**sheet, **meta}
                if updated["sheet_id"].startswith("UNKNOWN"):
                    # Only accept something shaped like a sheet number: on non-drawing pages the VLM
                    # returns page counters ("1/8") or "N/A", which also end up in file names.
                    vid = clean_sheet_id(meta.get("verified_sheet_id", ""))
                    if vid:
                        updated["sheet_id"] = vid
                if updated["sheet_title"] == "UNTITLED SHEET" and meta.get("verified_sheet_title"):
                    updated["sheet_title"] = meta["verified_sheet_title"].strip()

                return updated

            except errors.APIError as e:
                # Never retry 400-level client/schema errors (except 429 rate limits)
                if e.code != 429 and 400 <= e.code < 500:
                    log.error(f"{tag}: {page} rejected by Gemini ({e.code}): {e}")
                    raise
                if attempt == max_retries:
                    raise
                wait_sec = 2 ** attempt
                log.warning(f"{tag}: {page} retry {attempt}/{max_retries} in {wait_sec}s: {type(e).__name__}: {str(e)[:150]}")
                await asyncio.sleep(wait_sec)

            except Exception as e:
                if attempt == max_retries:
                    raise
                wait_sec = 2 ** attempt
                log.warning(f"{tag}: {page} retry {attempt}/{max_retries} in {wait_sec}s: {type(e).__name__}: {str(e)[:150]}")
                await asyncio.sleep(wait_sec)


def _unenriched(sheet: dict, err: Exception, tag: str = "") -> dict:
    """One bad page must not fail the whole file: keep it searchable by its CAD text alone.
    No `spatial_summary` key, so the resume logic retries it on the next run."""
    log.warning(f"{tag or sheet.get('version', '')}: page {sheet['page_number']} ({sheet['sheet_id']}) "
                f"skipped enrich, text-only: {type(err).__name__}: {str(err)[:150]}")
    return {**sheet, "enrich_error": str(err)[:300]}


async def enrich_manifest_async(
    version: str = "v1",
    output_dir: str = "./blueprint_data",
    max_concurrency: int = 10,
    semaphore: asyncio.Semaphore | None = None,
    client: genai.Client | None = None,
    on_progress: Callable[[], None] | None = None,
    tag: str = "",
):
    """Pass a shared `semaphore`/`client` to enrich many files at once against one API rate limit.
    `on_progress` is called once per finished page (for UI progress bars)."""
    start_time = time.perf_counter()
    client = client or genai.Client(
        api_key=os.environ["GEMINI_API_KEY"].strip(),
        http_options=HTTP_OPTIONS,
    )

    base_path = Path(output_dir) / version
    manifest_path = base_path / "manifest.json"
    enriched_path = base_path / "enriched_manifest.json"

    sheets = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Resume support: load any pages already enriched in a previous partial run
    existing_by_page: dict[int, dict] = {}
    if enriched_path.exists():
        for item in json.loads(enriched_path.read_text(encoding="utf-8")):
            existing_by_page[item["page_number"]] = item

    semaphore = semaphore or asyncio.Semaphore(max_concurrency)
    tasks = []

    for sheet in sheets:
        pnum = sheet["page_number"]
        if pnum in existing_by_page and "spatial_summary" in existing_by_page[pnum]:
            tasks.append(asyncio.sleep(0, result=existing_by_page[pnum]))
        else:
            tasks.append(enrich_single_sheet(client, sheet, semaphore, tag=tag))

    async def _tracked(coro, sheet):
        try:
            result = await coro
        except errors.APIError as e:
            if e.code != 429 and 400 <= e.code < 500:
                raise  # bad key/model/request: every page would fail, so fail loudly
            result = _unenriched(sheet, e, tag)
        except Exception as e:  # timeout, truncated/invalid JSON after all retries
            result = _unenriched(sheet, e, tag)
        if on_progress:
            on_progress()
        return result

    enriched_records = await asyncio.gather(*(_tracked(t, s) for t, s in zip(tasks, sheets)))

    # Sort by page order, deduplicate canonical_id, and rename any remaining UNKNOWN image files
    enriched_records.sort(key=lambda r: r["page_number"])
    seen_ids: dict[str, int] = {}

    for rec in enriched_records:
        sid = rec["sheet_id"]
        count = seen_ids.get(sid, 0) + 1
        seen_ids[sid] = count
        canonical_id = sid if count == 1 else f"{sid}_DUP{count}"
        rec["canonical_id"] = canonical_id

        old_img = Path(rec["image_path"])
        safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", canonical_id)  # defense in depth: always a valid file name
        new_img = old_img.with_name(f"p{rec['page_number']:03d}_{safe_id}.png")
        if old_img.exists() and old_img != new_img and not new_img.exists():
            old_img.rename(new_img)
            rec["image_path"] = str(new_img)

    enriched_path.write_text(json.dumps(enriched_records, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - start_time
    remaining_unknown = sum(1 for r in enriched_records if r["sheet_id"].startswith("UNKNOWN"))
    log.debug(f"{tag or version}: enriched {len(enriched_records)} sheets in {elapsed:.1f}s, "
              f"{len(enriched_records) - remaining_unknown} with a sheet ID")
    return enriched_records


if __name__ == "__main__":
    asyncio.run(enrich_manifest_async(version="v2", max_concurrency=10))