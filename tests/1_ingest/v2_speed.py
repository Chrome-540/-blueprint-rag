import hashlib
import json
import os
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


def read_pages(pdf_path: str, pages: list[int]) -> list[tuple[int, str]]:
    """(page number, text) of the given pages (numbers start at 1). Runs in a worker process."""
    with pymupdf.open(pdf_path) as doc:  # closed after each batch, so the PDF is not left locked
        return [(n, doc[n - 1].get_text("text").strip()) for n in pages]


def render_pages(pdf_path: str, jobs: list[tuple[int, str]], dpi: int = 200) -> None:
    """Save (page number, image path) pages as PNG, skipping existing ones. Runs in a worker process."""
    with pymupdf.open(pdf_path) as doc:
        for n, path in jobs:
            if not Path(path).exists():
                part = path + ".part"  # renamed only when complete: never a half-written image
                doc[n - 1].get_pixmap(dpi=dpi).save(part, output="png")
                os.replace(part, path)


def build_records(texts: list[tuple[int, str]], version: str, img_dir: Path) -> list[RawSheetRecord]:
    """One record per page, in page order (duplicate sheet numbers depend on that order)."""
    records: list[RawSheetRecord] = []
    seen_sheet_ids: dict[str, int] = {}
    for page_num, raw_cad_text in sorted(texts):
        sheet_id, sheet_title, rev_date = extract_title_block_fields(raw_cad_text, page_num)

        # Ensure canonical_id is unique even if a consultant duplicates a sheet number
        count = seen_sheet_ids.get(sheet_id, 0) + 1
        seen_sheet_ids[sheet_id] = count
        canonical_id = sheet_id if count == 1 else f"{sheet_id}_DUP{count}"

        # Deterministic SHA-256 hash of normalized CAD text
        normalized_text = " ".join(raw_cad_text.split())
        records.append(
            RawSheetRecord(
                version=version,
                page_number=page_num,
                sheet_id=sheet_id,
                canonical_id=canonical_id,
                sheet_title=sheet_title,
                revision_date=rev_date,
                text_hash=hashlib.sha256(normalized_text.encode("utf-8")).hexdigest(),
                image_path=str(img_dir / f"p{page_num:03d}_{canonical_id}.png"),
                raw_cad_text=raw_cad_text,
            )
        )
    return records


def save_manifest(records: list[RawSheetRecord], out_path: Path) -> None:
    (out_path / "manifest.json").write_text(
        json.dumps([r.model_dump() for r in records], indent=2), encoding="utf-8"
    )


def ingest_blueprint_pdf(
    pdf_path: str,
    version: str,
    output_dir: str = "./blueprint_data",
    dpi: int = 200,
) -> list[RawSheetRecord]:
    """Single-process version for the command line; the app runs the same steps in parallel (pipeline.py)."""
    out_path = Path(output_dir) / version
    img_dir = out_path / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(pdf_path) as doc:
        pages = list(range(1, doc.page_count + 1))
    records = build_records(read_pages(pdf_path, pages), version, img_dir)
    render_pages(pdf_path, [(r.page_number, r.image_path) for r in records], dpi)
    save_manifest(records, out_path)
    print(f"Saved {len(records)} pages -> {out_path / 'manifest.json'}")
    return records


if __name__ == "__main__":
    ingest_blueprint_pdf(
        pdf_path="Full Stamped Set For Construction 06-10-26 v2.pdf",
        version="v2",
    )