import asyncio
import json
import os
import time
from pathlib import Path
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from dotenv import load_dotenv

load_dotenv()
from google.genai import errors, types


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


async def enrich_single_sheet(
    client: genai.Client,
    sheet: dict,
    semaphore: asyncio.Semaphore,
    max_retries: int = 4,
) -> dict:
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
        for attempt in range(1, max_retries + 1):
            try:
                with Image.open(sheet["image_path"]) as img:
                    img.load()
                    response = await client.aio.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=[img, prompt],
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            response_schema=SheetSpatialMetadata,
                            temperature=0.1,
                        ),
                    )

                meta = json.loads(response.text)

                # Backfill Sheet ID & Title if Step 1 had marked them UNKNOWN / UNTITLED
                updated = {**sheet, **meta}
                if updated["sheet_id"].startswith("UNKNOWN") and meta.get("verified_sheet_id"):
                    updated["sheet_id"] = meta["verified_sheet_id"].strip().replace(" ", "")
                if updated["sheet_title"] == "UNTITLED SHEET" and meta.get("verified_sheet_title"):
                    updated["sheet_title"] = meta["verified_sheet_title"].strip()

                print(f"  [OK] Page {updated['page_number']:03d} | {updated['sheet_id']} - {updated['sheet_title'][:40]}")
                return updated

            except errors.APIError as e:
                # Never retry 400-level client/schema errors (except 429 rate limits)
                if e.code != 429 and 400 <= e.code < 500:
                    print(f"  [CLIENT ERROR {e.code}] Page {sheet['page_number']:03d}: {e}")
                    raise
                if attempt == max_retries:
                    print(f"  [FAILED] Page {sheet['page_number']:03d} ({sheet['sheet_id']}): {e}")
                    raise
                wait_sec = 2 ** attempt
                print(f"  [RETRY {attempt}/{max_retries}] Page {sheet['page_number']:03d} in {wait_sec}s -> {e}")
                await asyncio.sleep(wait_sec)

            except Exception as e:
                if attempt == max_retries:
                    print(f"  [FAILED] Page {sheet['page_number']:03d} ({sheet['sheet_id']}): {e}")
                    raise
                wait_sec = 2 ** attempt
                print(f"  [RETRY {attempt}/{max_retries}] Page {sheet['page_number']:03d} in {wait_sec}s -> {e}")
                await asyncio.sleep(wait_sec)


async def enrich_manifest_async(
    version: str = "v1",
    output_dir: str = "./blueprint_data",
    max_concurrency: int = 10,
):
    start_time = time.perf_counter()
    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"].strip(),
        http_options={"api_version": "v1beta"},
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

    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = []

    for sheet in sheets:
        pnum = sheet["page_number"]
        if pnum in existing_by_page and "spatial_summary" in existing_by_page[pnum]:
            tasks.append(asyncio.sleep(0, result=existing_by_page[pnum]))
        else:
            tasks.append(enrich_single_sheet(client, sheet, semaphore))

    print(f"Starting async enrichment for {len(sheets)} sheets (concurrency={max_concurrency})...")
    enriched_records = await asyncio.gather(*tasks)

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
        new_img = old_img.with_name(f"p{rec['page_number']:03d}_{canonical_id}.png")
        if old_img.exists() and old_img != new_img and not new_img.exists():
            old_img.rename(new_img)
            rec["image_path"] = str(new_img)

    enriched_path.write_text(json.dumps(enriched_records, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - start_time
    remaining_unknown = sum(1 for r in enriched_records if r["sheet_id"].startswith("UNKNOWN"))
    print(f"\nCompleted {len(enriched_records)} sheets in {elapsed:.2f}s -> {enriched_path}")
    print(f"Final identified sheets: {len(enriched_records) - remaining_unknown}/{len(enriched_records)}")


if __name__ == "__main__":
    asyncio.run(enrich_manifest_async(version="v2", max_concurrency=10))