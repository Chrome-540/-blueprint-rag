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
        description="The exact Sheet ID printed in the title block (e.g., 'A001', 'S202', 'M101'). Empty if you cannot read it."
    )
    verified_sheet_title: str = Field(
        description="The exact Sheet Title printed in the title block. Empty if you cannot read it."
    )
    discipline: str = Field(
        description="Primary discipline: 'Architectural', 'Life Safety', 'Civil', 'Structural', 'Mechanical', 'Electrical', 'Plumbing', 'Fire Protection', or 'Unknown'."
    )
    floor_level: str = Field(
        description="Floor or scope stated on the sheet, e.g. 'Site', 'Foundation', '1st Floor', 'Roof', 'Details/Sections', or 'Unknown'."
    )
    grid_bounds: str = Field(
        description="Structural grid range, only if the grid labels are readable (e.g., 'Horizontal 1-25, Vertical A-S'), else 'N/A'."
    )
    referenced_sheets: list[str] = Field(
        default_factory=list,
        description="Other sheet IDs that appear in callouts or notes on this page. Empty if none."
    )
    key_entities: list[str] = Field(
        default_factory=list,
        description="At most 25 important room names/numbers, members, equipment or code sections, copied exactly as printed."
    )
    spatial_summary: str = Field(
        description="What this sheet shows, in at most 5 sentences. Fewer is fine. Only what is readable or clearly visible."
    )


GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
# 2.5 Flash thinks (and bills for it) unless told not to; 3.5 Flash-Lite rejects that setting with a 400.
NO_THINKING = types.ThinkingConfig(thinking_budget=0) if GEMINI_MODEL.startswith("gemini-2.5-flash") else None
MAX_TEXT_CHARS = 30000
USAGE = {"calls": 0, "in": 0, "out": 0}
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
You are describing one construction drawing sheet so that it can be FOUND by search.
You get the sheet image and the exact text stored in the PDF for this page.

Sheet ID: {sheet['sheet_id']}
Sheet title: {sheet['sheet_title']}

PDF text of this page:
{sheet['raw_cad_text'][:MAX_TEXT_CHARS]}

Rules:
1. Wrong information is worse than no information. Leave out anything you are not sure about.
2. Names, numbers, sizes and room names must appear in the PDF text above. Copy them exactly, never change or invent them.
3. Use the image to understand what kind of drawing this is and how it is laid out. Do not name objects (furniture,
   fixtures, equipment) unless they are labelled.
4. Do not count things, measure things, or draw conclusions.
5. If something is not on the sheet, leave it empty.
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
                        thinking_config=NO_THINKING,
                        # The default gives a whole 36x24 in sheet only ~260 tokens on 2.5 Flash; HIGH gave ~3,350.
                        media_resolution=types.MediaResolution.MEDIA_RESOLUTION_HIGH,
                    ),
                )

                if u := response.usage_metadata:
                    USAGE["calls"] += 1
                    USAGE["in"] += u.prompt_token_count or 0
                    USAGE["out"] += u.candidates_token_count or 0
                meta = json.loads(response.text)

                # Backfill Sheet ID & Title if Step 1 had marked them UNKNOWN / UNTITLED
                updated = {**sheet, **meta}
                if updated["sheet_id"].startswith("UNKNOWN"):
                    # Only accept something shaped like a sheet number: on non-drawing pages the VLM
                    # returns page counters ("1/8") or "N/A", which also end up in file names.
                    vid = clean_sheet_id(meta.get("verified_sheet_id", ""))
                    plain = re.sub(r"[-.]", "", sheet["raw_cad_text"].upper()).split()
                    if vid and re.sub(r"[-.]", "", vid) in plain:  # must really be printed on this page
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