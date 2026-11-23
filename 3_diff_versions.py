"""Compare two versions of a drawing set. No AI: every fact comes from the PDFs' own text and pixels.

Per sheet: text added / removed / in a new place (exact, from the PDF text layer) and a red/green
overlay diffs/diff_v1_to_v2_<SHEET>.png (red = only in v1, green = only in v2, gray = unchanged).

Needs only the output of ingest (manifest.json + images). Anything that cannot be compared safely is
NEEDS_REVIEW, never guessed.
CLI: python 3_diff_versions.py --selftest
"""
import asyncio
import json
import logging
import os
import re
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pymupdf

log = logging.getLogger("blueprint")
FORMAT = 2  # of diff_summary_*.json; older files were written by the AI version

# ponytail: calibration knobs. Lower = catches smaller changes but more noise.
TOLERANCE_PX = int(os.getenv("DIFF_TOLERANCE_PX", "1"))      # ink this close counts as unchanged
MIN_INK_PX = int(os.getenv("DIFF_MIN_INK_PX", "20"))         # smaller specks are ignored
TEXT_TOLERANCE_PX = float(os.getenv("DIFF_TEXT_TOLERANCE_PX", "3"))
# ponytail: title block assumed to be the strip at the right edge; adjust for other layouts.
TITLE_BLOCK_X = float(os.getenv("DIFF_TITLE_BLOCK_X", "0.935"))
WORKERS = int(os.getenv("DIFF_WORKERS", "4"))
CELL = 8              # change map resolution, px
REPEAT_SHARE = 0.6    # changed at the same spot on this share of sheets = issue stamp / seal, not drawing
INK = 235             # darker than this is ink (keeps light gray linework)


def _near(ink: np.ndarray) -> np.ndarray:
    k = 2 * TOLERANCE_PX + 1
    return cv2.dilate(ink.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)


def _overlap(i1: np.ndarray, i2: np.ndarray) -> float:
    """Share of the ink both sheets have in common (0..1)."""
    common = np.count_nonzero(i1 & _near(i2)) + np.count_nonzero(i2 & _near(i1))
    return common / max(1, np.count_nonzero(i1) + np.count_nonzero(i2))


def align(img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray | None, np.ndarray]:
    """(img1 moved onto img2, 2x3 transform). Image is None if the sheets cannot be lined up."""
    identity = np.float64([[1, 0, 0], [0, 1, 0]])
    same_size = img1.shape == img2.shape
    base = _overlap(img1 < INK, img2 < INK) if same_size else 0.0
    best, transform = (img1 if same_size else None), identity
    if base < 0.97:
        orb = cv2.ORB_create(5000)
        k1, d1 = orb.detectAndCompute(img1, None)
        k2, d2 = orb.detectAndCompute(img2, None)
        if d1 is not None and d2 is not None:
            m = sorted(cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d1, d2), key=lambda x: x.distance)[:500]
            if len(m) >= 30:
                src = np.float32([k1[x.queryIdx].pt for x in m])
                dst = np.float32([k2[x.trainIdx].pt for x in m])
                found, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3)
                if found is not None and inliers.sum() >= 30:
                    h, w = img2.shape
                    corners = np.float64([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]]).T
                    if not same_size or np.abs(found @ corners - identity @ corners).max() > 0.5:
                        warped = cv2.warpAffine(img1, found, (w, h), borderValue=255)
                        score = _overlap(warped < INK, img2 < INK)
                        if score > base:  # never replace a better raw fit
                            best, transform, base = warped, found, score
    # A wrong fit would paint everything red/green: better to say "could not compare".
    return (best, transform) if base >= 0.5 else (None, identity)


def visual_diff(img1: np.ndarray, img2: np.ndarray, overlay_path: Path) -> np.ndarray:
    """Writes the overlay (if anything differs) and returns changed ink per CELL x CELL square."""
    i1, i2 = img1 < INK, img2 < INK
    removed, added = i1 & ~_near(i2), i2 & ~_near(i1)
    changed = removed | added
    h, w = i2.shape
    ch, cw = -(-h // CELL), -(-w // CELL)
    padded = np.zeros((ch * CELL, cw * CELL), np.uint8)
    padded[:h, :w] = changed
    cells = padded.reshape(ch, CELL, cw, CELL).sum(axis=(1, 3), dtype=np.uint16)
    if changed.any():
        canvas = np.full((h, w, 3), 255, np.uint8)
        canvas[i1 | i2] = (210, 210, 210)
        for ink, pale in ((removed, (170, 170, 255)), (added, (170, 235, 170))):  # thin lines stay visible when shrunk
            canvas[cv2.dilate(ink.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool) & ~(i1 | i2)] = pale
        canvas[removed] = (0, 0, 220)
        canvas[added] = (0, 170, 0)
        overlay_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(overlay_path), canvas)
    return cells


def text_lines(pdf: str, page_number: int, size: tuple[int, int]) -> list[tuple[str, float, float]]:
    """(text, x, y) of every text line of the page, in pixels of its rendered image (w, h)."""
    out = []
    with pymupdf.open(pdf) as doc:
        page = doc[page_number - 1]
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = re.sub(r"\s+", " ", "".join(s["text"] for s in line["spans"])).strip()
                if text:
                    r = pymupdf.Rect(line["bbox"]) * page.rotation_matrix
                    out.append((text, (r.x0 + r.x1) / 2 / page.rect.width * size[0],
                                (r.y0 + r.y1) / 2 / page.rect.height * size[1]))
    return out


def text_diff(old: list, new: list, transform: np.ndarray, width: int) -> tuple[dict, dict]:
    """({added, removed, relocated} in the drawing area, same for the title block strip)."""
    by_text = defaultdict(list)
    for j, (t, x, y) in enumerate(new):
        by_text[t].append((j, x, y))
    used, left = set(), []
    for t, x, y in old:
        x, y = transform @ (x, y, 1)
        j = next((j for j, p, q in by_text[t] if j not in used and abs(p - x) <= TEXT_TOLERANCE_PX
                  and abs(q - y) <= TEXT_TOLERANCE_PX), None)
        if j is None:
            left.append((t, x))
        else:
            used.add(j)
    drawing, title = ({"added": [], "removed": [], "relocated": []} for _ in range(2))

    def put(kind: str, t: str, x: float) -> None:
        (title if x > TITLE_BLOCK_X * width else drawing)[kind].append(t)

    for t, x in left:
        j, p = next(((j, p) for j, p, _ in by_text[t] if j not in used
                     and (p > TITLE_BLOCK_X * width) == (x > TITLE_BLOCK_X * width)), (None, 0))
        if j is None:
            put("removed", t, x)
        else:
            used.add(j)
            put("relocated", t, p)
    for j, (t, x, _) in enumerate(new):
        if j not in used:
            put("added", t, x)
    return drawing, title


def compare_sheet(pdf1: str, pdf2: str, s1: dict, s2: dict, overlay_path: str) -> dict:
    """Runs in a worker process."""
    img1 = cv2.imread(s1["image_path"], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(s2["image_path"], cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        return {"problem": "a sheet image is missing"}
    size1, size2 = img1.shape[::-1], img2.shape[::-1]
    img1, transform = align(img1, img2)
    if img1 is None:
        return {"problem": "the drawing was moved or redrawn too much to line up"}
    cells = visual_diff(img1, img2, Path(overlay_path))
    drawing, title = text_diff(text_lines(pdf1, s1["page_number"], size1), text_lines(pdf2, s2["page_number"], size2),
                               transform, size2[0])
    return {"cells": cells, "text": drawing, "title_text": title, "overlay": overlay_path if cells.any() else None}


def repeated_spots(cell_maps: list[np.ndarray]) -> np.ndarray | None:
    """Squares that changed on most sheets (issue stamp, seal). None for small or mixed-size sets."""
    if len(cell_maps) < 8 or len({c.shape for c in cell_maps}) != 1:
        return None
    share = np.mean([c > 0 for c in cell_maps], axis=0)
    spots = cv2.dilate((share >= REPEAT_SHARE).astype(np.uint8), np.ones((5, 5), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(spots)
    for x, y, w, h, _ in stats[1:]:  # a stamp is a box: its letters shift a little from sheet to sheet
        if w * h <= 0.01 * spots.size:  # never a big box, that would hide real changes
            spots[y:y + h, x:x + w] = 1
    return spots.astype(bool)


def drawing_areas(cells: np.ndarray, repeated: np.ndarray | None) -> int:
    """Number of separate changed areas in the drawing (outside title block and stamps)."""
    c = cells.copy()
    c[:, int(c.shape[1] * TITLE_BLOCK_X):] = 0
    if repeated is not None:
        c[repeated] = 0
    n, labels = cv2.connectedComponents(cv2.dilate((c > 0).astype(np.uint8), np.ones((9, 9), np.uint8)))
    ink = np.bincount(labels.ravel(), weights=c.ravel(), minlength=n)
    return int(np.count_nonzero(ink[1:] >= MIN_INK_PX))


def printed(sheet_id: str, sheet: dict) -> bool:
    """True if the sheet number is really printed in the page's own text."""
    plain = lambda t: re.sub(r"[-.]", "", t.upper())  # noqa: E731
    return plain(sheet_id) in plain(sheet["raw_cad_text"]).split()


async def diff_versions_async(
    pdf1: str,
    pdf2: str,
    sheets1: list[dict],
    sheets2: list[dict],
    out_dir: str,
    labels: tuple[str, str] = ("v1", "v2"),
    on_progress: Callable[[], None] | None = None,
    tag: str = "compare",
) -> dict:
    """Compare every sheet of the new set (pdf2, sheets2) with the old one. `sheets` are ingest records."""
    t0 = time.perf_counter()
    prev, curr = labels
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)

    def by_number(sheets: list[dict]) -> dict[str, list[dict]]:
        by_id = defaultdict(list)
        for s in sheets:
            by_id[s["sheet_id"]].append(s)
        return by_id

    old, new = by_number(sheets1), by_number(sheets2)
    loop = asyncio.get_running_loop()

    async def one(sid: str, pool) -> dict:
        a, b = old.get(sid, []), new.get(sid, [])
        entry = {"sheet_id": sid, "sheet_title": (b or a)[0]["sheet_title"], "overlay": None, "problem": "",
                 "text": {}, "title_text": {}, "areas": 0,
                 "prev": [{k: s[k] for k in ("page_number", "image_path")} for s in a],
                 "curr": [{k: s[k] for k in ("page_number", "image_path")} for s in b]}
        if sid.startswith("UNKNOWN"):
            entry.update(status="NEEDS_REVIEW", problem="no sheet number could be read")
        elif not all(printed(sid, s) for s in a + b):  # AI helps read numbers: pair only on confirmed ones
            entry.update(status="NEEDS_REVIEW", problem="the sheet number could not be confirmed in the PDF text")
        elif not a:
            entry["status"] = "NEW_SHEET"
        elif not b:
            entry["status"] = "DELETED"
        elif len(a) > 1 or len(b) > 1:  # pairing copies would be a guess
            entry.update(status="NEEDS_REVIEW", problem="this sheet number is used more than once in a set")
        else:
            safe = re.sub(r"[^A-Za-z0-9._-]", "_", sid)
            overlay = base / "diffs" / f"diff_{prev}_to_{curr}_{safe}.png"
            overlay.unlink(missing_ok=True)
            try:
                r = await loop.run_in_executor(pool, compare_sheet, pdf1, pdf2, a[0], b[0], str(overlay))
            except Exception as e:  # one bad sheet must not fail the comparison
                log.warning(f"{tag}: {sid} failed: {type(e).__name__}: {str(e)[:150]}")
                r = {"problem": "the comparison of this sheet failed"}
            entry.update(r)
            entry["status"] = "NEEDS_REVIEW" if r.get("problem") else None  # decided below
        if on_progress:
            on_progress()
        return entry

    ids = list(new) + [i for i in old if i not in new]
    with ProcessPoolExecutor(max_workers=max(1, WORKERS)) as pool:
        sheets = await asyncio.gather(*(one(i, pool) for i in ids))

    repeated = repeated_spots([s["cells"] for s in sheets if "cells" in s])
    for s in sheets:
        cells = s.pop("cells", None)
        if s["status"] is None:
            s["areas"] = drawing_areas(cells, repeated)
            if s["areas"] or any(s["text"].values()):
                s["status"] = "MODIFIED"
            else:
                s["status"] = "TITLE_ONLY" if s["overlay"] or any(s["title_text"].values()) else "UNCHANGED"

    counts = Counter(s["status"] for s in sheets)
    result = {"format": FORMAT, "prev_version": prev, "curr_version": curr, "counts": dict(counts), "sheets": sheets}
    out = base / f"diff_summary_{prev}_to_{curr}.json"
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"{tag}: done ({counts['MODIFIED']} changed, {counts['TITLE_ONLY']} title block only, "
             f"{counts['UNCHANGED']} unchanged, {counts['NEW_SHEET']} added, {counts['DELETED']} deleted, "
             f"{counts['NEEDS_REVIEW']} need review, {time.perf_counter() - t0:.0f}s)")
    return result


def _selftest() -> None:
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)
    page = np.full((1600, 2400), 255, np.uint8)
    for x, y in rng.integers(100, 1300, (150, 2)):
        cv2.rectangle(page, (int(x), int(y)), (int(x) + int(rng.integers(20, 200)), int(y) + int(rng.integers(20, 200))), 0, 2)

    def areas(a, b):
        fitted, _ = align(a, b)
        assert fitted is not None
        return drawing_areas(visual_diff(fitted, b, tmp / "o.png"), None)

    assert areas(page, page) == 0
    for s in (1, 2, 5, 40):  # same drawing, shifted on the page
        assert areas(np.roll(page, (s, s), (0, 1)), page) == 0, s
    door = page.copy()
    cv2.ellipse(door, (1800, 800), (60, 60), 0, 0, 90, 0, 2)
    assert areas(page, door) == 1 and areas(door, page) == 1
    dot = page.copy()
    cv2.circle(dot, (1800, 800), 6, 0, -1)
    assert areas(page, dot) == 1
    stamp = page.copy()
    cv2.rectangle(stamp, (2300, 100), (2380, 200), 0, 3)  # title block strip
    assert areas(page, stamp) == 0
    other = np.full_like(page, 255)
    for x, y in rng.integers(100, 1300, (150, 2)):
        cv2.circle(other, (int(x), int(y)), int(rng.integers(5, 60)), 0, 2)
    assert align(page, other)[0] is None  # different drawing: refuse, don't paint

    eye = np.float64([[1, 0, 0], [0, 1, 0]])
    d, t = text_diff([("A", 10, 10), ("B", 50, 50), ("C", 90, 90), ("D1", 990, 5)],
                     [("A", 11, 10), ("B", 80, 50), ("E", 90, 90), ("D2", 990, 5)], eye, 1000)
    assert d == {"added": ["E"], "removed": ["C"], "relocated": ["B"]}
    assert t == {"added": ["D2"], "removed": ["D1"], "relocated": []}
    maps = [np.zeros((200, 300), np.uint16) for _ in range(10)]
    for k, m in enumerate(maps):
        m[180, 200:220] = m[190, 200:220] = 30  # stamp box
        m[180:191, 200] = m[180:191, 219] = 30
        m[185, 205 + k % 5] = 30                # letters, different on each sheet
    maps[0][20, 20] = 30
    rep = repeated_spots(maps)
    assert drawing_areas(maps[0], rep) == 1 and drawing_areas(maps[1], rep) == 0  # stamp ignored, real change kept
    frame = [np.zeros((200, 300), np.uint16) for _ in range(10)]
    for m in frame:
        m[10, 10:270] = m[190, 10:270] = m[10:191, 10] = m[10:191, 269] = 30  # sheet border on every sheet
    frame[0][100, 100] = 30
    assert drawing_areas(frame[0], repeated_spots(frame)) == 1  # a big box must not hide what's inside
    assert printed("A101", {"raw_cad_text": "SHEET: A-101"}) and not printed("A101", {"raw_cad_text": "A1011 B-25"})
    print("DIFF SELFTEST OK")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--selftest"]:
        _selftest()
    else:
        sys.exit("Use tests/3_diff/run.py")
