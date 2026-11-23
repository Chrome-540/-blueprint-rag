import asyncio
import difflib
import json
import os
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
from google import genai
from dotenv import load_dotenv

load_dotenv()
from google.genai import errors, types


class SheetDiffAnalysis(BaseModel):
    has_meaningful_changes: bool = Field(
        description="True if there are actual architectural, structural, MEP, schedule, or revision-cloud changes. False if only minor date/title-block formatting shifted."
    )
    change_summary: str = Field(
        description="Concise bulleted engineering summary of every revision added, removed, or modified between the two versions (including grid locations, room numbers, or member sizes)."
    )


def compute_visual_diff(
    img_v1_path: str,
    img_v2_path: str,
    overlay_out_path: Path,
    pixel_threshold: int = 25,
) -> tuple[float, bool]:
    """
    Aligns v1 and v2 blueprint PNGs, calculates the percentage of changed pixels,
    and saves a visual red/green diff overlay (Red = removed in v1, Green = added in v2).
    """
    img1 = cv2.imread(img_v1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_v2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        return 1.0, False

    # Resize img1 to match img2 if page dimensions shifted slightly
    if img1.shape != img2.shape:
        img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]), interpolation=cv2.INTER_AREA)

    diff = cv2.absdiff(img1, img2)
    _, thresh = cv2.threshold(diff, pixel_threshold, 255, cv2.THRESH_BINARY)

    changed_pixels = int(np.count_nonzero(thresh))
    total_pixels = thresh.size
    changed_ratio = changed_pixels / max(total_pixels, 1)

    # Generate visual overlay only if there is non-trivial pixel difference (> 0.02%)
    if changed_ratio > 0.0002:
        overlay_out_path.parent.mkdir(parents=True, exist_ok=True)
        # White background with faint gray unchanged lines; Red = only in v1, Green = only in v2
        h, w = img2.shape
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)

        v1_ink = img1 < 200
        v2_ink = img2 < 200
        both_ink = v1_ink & v2_ink
        removed_ink = v1_ink & (~v2_ink)
        added_ink = v2_ink & (~v1_ink)

        canvas[both_ink] = [210, 210, 210]   # Faint gray for unchanged drawing
        canvas[removed_ink] = [0, 0, 220]    # Red (BGR) for deleted elements
        canvas[added_ink] = [0, 170, 0]      # Green (BGR) for newly added elements

        # Dilate colored diff pixels slightly so thin CAD lines/clouds pop clearly for VLM & humans
        mask = (removed_ink | added_ink).astype(np.uint8) * 255
        dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
        # Paint dilated halo where canvas is still white
        halo = (dilated > 0) & (~both_ink) & (~removed_ink) & (~added_ink)
        canvas[halo] = [255, 220, 180]       # Soft blue-tint highlight around changes

        cv2.imwrite(str(overlay_out_path), canvas)
        return changed_ratio, True

    return changed_ratio, False


def build_text_delta(text_v1: str, text_v2: str, max_chars: int = 6000) -> str:
    """Extracts added and removed lines between v1 and v2 CAD text."""
    lines1 = [l.strip() for l in text_v1.splitlines() if l.strip()]
    lines2 = [l.strip() for l in text_v2.splitlines() if l.strip()]
    delta_lines = [
        line for line in difflib.unified_diff(lines1, lines2, lineterm="", n=1)
        if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]
    return "\n".join(delta_lines)[:max_chars]


async def analyze_changed_sheet(
    client: genai.Client,
    prev_version: str,
    curr_version: str,
    v1_sheet: dict,
    v2_sheet: dict,
    overlay_path: str | None,
    text_delta: str,
    semaphore: asyncio.Semaphore,
    max_retries: int = 4,
) -> dict:
    async with semaphore:
        prompt = f"""
You are a senior construction & architectural QA engineer comparing two revisions of the same blueprint sheet.
Previous Version: {prev_version} | Current Version: {curr_version}
Sheet ID: {v2_sheet['sheet_id']} | Title: {v2_sheet['sheet_title']}

You are given:
1. Image 1: Previous version ({prev_version})
2. Image 2: Current version ({curr_version})
3. Image 3 (if attached): Visual Diff Overlay (Red = removed from {prev_version}, Green = added in {curr_version})

CAD Text Differences (+ added, - removed):
{text_delta or "(No raw text lines changed; changes are purely graphical/vector)"}

Inspect the revision clouds, delta symbols, schedules, dimensions, and layout differences.
Return `has_meaningful_changes` and a crisp engineering `change_summary`.
"""
        for attempt in range(1, max_retries + 1):
            try:
                images = []
                with Image.open(v1_sheet["image_path"]) as im1:
                    im1.load()
                    images.append(im1.copy())
                with Image.open(v2_sheet["image_path"]) as im2:
                    im2.load()
                    images.append(im2.copy())
                if overlay_path and Path(overlay_path).exists():
                    with Image.open(overlay_path) as im_ov:
                        im_ov.load()
                        images.append(im_ov.copy())

                response = await client.aio.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=[*images, prompt],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=SheetDiffAnalysis,
                        temperature=0.1,
                    ),
                )

                analysis = json.loads(response.text)
                diff_record = {
                    "compared_against": prev_version,
                    "status": "MODIFIED" if analysis["has_meaningful_changes"] else "UNCHANGED_MINOR",
                    "has_meaningful_changes": analysis["has_meaningful_changes"],
                    "change_summary": analysis["change_summary"],
                    "diff_overlay_path": overlay_path if analysis["has_meaningful_changes"] else None,
                }
                print(f"  [DIFF] {v2_sheet['canonical_id']} -> {diff_record['status']}")
                return {**v2_sheet, "diff_from_prev": diff_record}

            except errors.APIError as e:
                if e.code != 429 and 400 <= e.code < 500:
                    raise
                if attempt == max_retries:
                    raise
                await asyncio.sleep(2 ** attempt)
            except Exception:
                if attempt == max_retries:
                    raise
                await asyncio.sleep(2 ** attempt)


async def diff_versions_async(
    prev_version: str = "v1",
    curr_version: str = "v2",
    output_dir: str = "./blueprint_data",
    max_concurrency: int = 8,
):
    start_time = time.perf_counter()
    base = Path(output_dir)
    v1_path = base / prev_version / "enriched_manifest.json"
    v2_path = base / curr_version / "enriched_manifest.json"
    diff_dir = base / curr_version / "diffs"

    v1_sheets = json.loads(v1_path.read_text(encoding="utf-8"))
    v2_sheets = json.loads(v2_path.read_text(encoding="utf-8"))

    # Index v1 sheets by canonical_id (with fallback lookup by sheet_id)
    v1_by_canonical = {s["canonical_id"]: s for s in v1_sheets}
    v1_by_sheet_id = {}
    for s in v1_sheets:
        v1_by_sheet_id.setdefault(s["sheet_id"], s)

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"].strip(),
        http_options={"api_version": "v1beta"},
    )
    semaphore = asyncio.Semaphore(max_concurrency)
    tasks = []
    unchanged_count = 0
    new_sheet_count = 0

    print(f"Comparing '{curr_version}' ({len(v2_sheets)} sheets) against '{prev_version}' ({len(v1_sheets)} sheets)...")

    for v2_s in v2_sheets:
        cid = v2_s["canonical_id"]
        v1_s = v1_by_canonical.get(cid) or v1_by_sheet_id.get(v2_s["sheet_id"])

        # Case 1: Brand new sheet added in v2
        if v1_s is None:
            new_sheet_count += 1
            diff_record = {
                "compared_against": prev_version,
                "status": "NEW_SHEET",
                "has_meaningful_changes": True,
                "change_summary": f"New sheet '{v2_s['sheet_id']} - {v2_s['sheet_title']}' added in {curr_version}.",
                "diff_overlay_path": None,
            }
            tasks.append(asyncio.sleep(0, result={**v2_s, "diff_from_prev": diff_record}))
            continue

        # Case 2: Check text hash + OpenCV pixel diff
        overlay_file = diff_dir / f"diff_{prev_version}_to_{curr_version}_{cid}.png"
        changed_ratio, has_overlay = compute_visual_diff(
            v1_s["image_path"], v2_s["image_path"], overlay_file
        )
        text_changed = v1_s.get("text_hash") != v2_s.get("text_hash")

        if not text_changed and not has_overlay:
            unchanged_count += 1
            diff_record = {
                "compared_against": prev_version,
                "status": "UNCHANGED",
                "has_meaningful_changes": False,
                "change_summary": "No changes from previous version.",
                "diff_overlay_path": None,
            }
            tasks.append(asyncio.sleep(0, result={**v2_s, "diff_from_prev": diff_record}))
        else:
            text_delta = build_text_delta(v1_s["raw_cad_text"], v2_s["raw_cad_text"])
            tasks.append(
                analyze_changed_sheet(
                    client=client,
                    prev_version=prev_version,
                    curr_version=curr_version,
                    v1_sheet=v1_s,
                    v2_sheet=v2_s,
                    overlay_path=str(overlay_file) if has_overlay else None,
                    text_delta=text_delta,
                    semaphore=semaphore,
                )
            )

    updated_v2 = await asyncio.gather(*tasks)
    updated_v2.sort(key=lambda r: r["page_number"])
    v2_path.write_text(json.dumps(updated_v2, indent=2), encoding="utf-8")

    # Also record any sheets from v1 that were deleted in v2
    v2_sheet_ids = {s["sheet_id"] for s in v2_sheets}
    deleted_sheets = [
        {"sheet_id": s["sheet_id"], "sheet_title": s["sheet_title"], "page_number": s["page_number"]}
        for s in v1_sheets if s["sheet_id"] not in v2_sheet_ids
    ]
    summary_path = base / curr_version / f"diff_summary_{prev_version}_to_{curr_version}.json"
    summary_path.write_text(
        json.dumps(
            {
                "prev_version": prev_version,
                "curr_version": curr_version,
                "unchanged_sheets": unchanged_count,
                "new_sheets": new_sheet_count,
                "deleted_sheets": deleted_sheets,
                "modified_sheets": [
                    {
                        "sheet_id": s["sheet_id"],
                        "sheet_title": s["sheet_title"],
                        "change_summary": s["diff_from_prev"]["change_summary"],
                        "diff_overlay_path": s["diff_from_prev"]["diff_overlay_path"],
                    }
                    for s in updated_v2
                    if s["diff_from_prev"]["status"] == "MODIFIED"
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - start_time
    print(f"\nDiff complete in {elapsed:.2f}s -> {summary_path}")
    print(f"Unchanged: {unchanged_count} | New: {new_sheet_count} | Deleted: {len(deleted_sheets)}")


if __name__ == "__main__":
    asyncio.run(diff_versions_async(prev_version="v1", curr_version="v2"))