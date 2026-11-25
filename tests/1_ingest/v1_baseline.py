import hashlib
import json
import re
from pathlib import Path
import pymupdf  # PyMuPDF
from pydantic import BaseModel


class RawSheetRecord(BaseModel):
    version: str
    page_number: int
    sheet_id: str          # e.g., "A001", "LS101", "S101", "S202"
    canonical_id: str      # Unique key within a version
    sheet_title: str
    revision_date: str | None
    text_hash: str         # SHA-256 of normalized CAD text for instant diff checks
    image_path: str
    raw_cad_text: str


# Matches A001, LS101, S101, S202, A403, S-202, M1.01, FP101A, etc.
SHEET_ID_PATTERN = re.compile(r"^[A-Z]{1,3}[-.]?\d{2,3}(?:\.\d+)?[A-Z]?$")


def extract_title_block_fields(raw_cad_text: str, page_num: int) -> tuple[str, str, str | None]:
    """
    Finds the title block anchor ('DATE :' followed by 'JOB NO. :') anywhere in raw_cad_text,
    extracting the Sheet ID & Title immediately preceding 'DATE :', plus Revision Date.
    """
    lines = [line.strip() for line in raw_cad_text.splitlines() if line.strip()]

    sheet_id = f"UNKNOWN_P{page_num:03d}"
    sheet_title = "UNTITLED SHEET"
    rev_date = None

    # 1. Locate the ('DATE :', 'JOB NO. :') title block anchor anywhere in the page's text stream
    anchor_idx = None
    for i in range(len(lines) - 1):
        if lines[i].upper().startswith("DATE") and lines[i + 1].upper().startswith("JOB NO"):
            anchor_idx = i
            break

    if anchor_idx is not None:
        # Look up to 6 lines immediately BEFORE 'DATE :' for [Sheet ID, Title Line 1, Title Line 2...]
        before_block = lines[max(0, anchor_idx - 6):anchor_idx]
        for idx, val in enumerate(before_block):
            cleaned = val.replace(" ", "")
            if SHEET_ID_PATTERN.fullmatch(cleaned):
                sheet_id = cleaned
                title_parts = [
                    p for p in before_block[idx + 1:]
                    if p.replace(" ", "") != sheet_id
                    and not p.upper().startswith(("SCALE:", "SHEET SIZE", "REVISIONS"))
                ]
                if title_parts:
                    sheet_title = " ".join(title_parts)
                break

        # Extract base date (e.g. '03/18/26') right after 'DATE :'
        after_block = lines[anchor_idx:anchor_idx + 15]
        for val in after_block:
            if re.fullmatch(r"\d{2}[/-]\d{2}[/-]\d{2,4}", val):
                rev_date = val
                break

    # 2. Cover sheet / revision block fallback (e.g., Page 1 'A001' sits right above '#1 06-08-26')
    for i, line in enumerate(lines):
        rev_match = re.search(r"#\d+\s+(\d{2}[/-]\d{2}[/-]\d{2,4})", line)
        if rev_match:
            rev_date = rev_match.group(1)
            if sheet_id.startswith("UNKNOWN") and i > 0:
                prev_cleaned = lines[i - 1].replace(" ", "")
                if SHEET_ID_PATTERN.fullmatch(prev_cleaned):
                    sheet_id = prev_cleaned
                    if sheet_title == "UNTITLED SHEET" and i >= 2:
                        sheet_title = lines[i - 2]
            break

    # 3. Last-resort fallback for consultant sheets without 'DATE :' + 'JOB NO. :'
    #    Check lines immediately following 'SHEET:' anywhere in the document
    if sheet_id.startswith("UNKNOWN"):
        for i, line in enumerate(lines):
            if line.upper().startswith("SHEET:") and i + 1 < len(lines):
                candidate = lines[i + 1].replace(" ", "")
                if SHEET_ID_PATTERN.fullmatch(candidate):
                    sheet_id = candidate
                    break

    return sheet_id, sheet_title, rev_date


def ingest_blueprint_pdf(
    pdf_path: str,
    version: str,
    output_dir: str = "./blueprint_data",
    dpi: int = 200,
) -> list[RawSheetRecord]:
    out_path = Path(output_dir) / version
    img_dir = out_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)
    records: list[RawSheetRecord] = []
    seen_sheet_ids: dict[str, int] = {}

    print(f"Ingesting '{pdf_path}' ({total_pages} pages) as version '{version}'...")

    for i, page in enumerate(doc):
        page_num = i + 1
        raw_cad_text = page.get_text("text").strip()
        sheet_id, sheet_title, rev_date = extract_title_block_fields(raw_cad_text, page_num)

        # Ensure canonical_id is unique even if a consultant duplicates a sheet number
        count = seen_sheet_ids.get(sheet_id, 0) + 1
        seen_sheet_ids[sheet_id] = count
        canonical_id = sheet_id if count == 1 else f"{sheet_id}_DUP{count}"

        # Deterministic SHA-256 hash of normalized CAD text
        normalized_text = " ".join(raw_cad_text.split())
        text_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()

        img_filename = f"p{page_num:03d}_{canonical_id}.png"
        img_filepath = img_dir / img_filename
        if not img_filepath.exists():
            pix = page.get_pixmap(dpi=dpi)
            pix.save(str(img_filepath))

        records.append(
            RawSheetRecord(
                version=version,
                page_number=page_num,
                sheet_id=sheet_id,
                canonical_id=canonical_id,
                sheet_title=sheet_title,
                revision_date=rev_date,
                text_hash=text_hash,
                image_path=str(img_filepath),
                raw_cad_text=raw_cad_text,
            )
        )

        if page_num % 25 == 0 or page_num == total_pages:
            print(f"  Processed {page_num}/{total_pages} pages...")

    manifest_path = out_path / "manifest.json"
    manifest_path.write_text(
        json.dumps([r.model_dump() for r in records], indent=2),
        encoding="utf-8",
    )

    unknown_count = sum(1 for r in records if r.sheet_id.startswith("UNKNOWN"))
    dup_count = sum(1 for r in records if "_DUP" in r.canonical_id)
    print(f"\nDone! Saved {len(records)} records -> {manifest_path}")
    print(f"Parsed: {len(records) - unknown_count}/{len(records)} | Duplicates: {dup_count}")
    return records


if __name__ == "__main__":
    ingest_blueprint_pdf(
        pdf_path="Full Stamped Set For Construction 06-10-26 v2.pdf",
        version="v2",
    )