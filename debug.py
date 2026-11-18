import json
from pathlib import Path

MANUAL_OVERRIDES = {
    ("v1", 50): ("A409", "DUMPSTER PLAN AND SECTION"),
    ("v2", 51): ("A409", "DUMPSTER PLAN AND SECTION"),
    ("v2", 80): ("A520", "INTERIOR ELEVATIONS - GUESTROOMS"),
}

for ver in ["v1", "v2"]:
    path = Path(f"blueprint_data/{ver}/enriched_manifest.json")
    records = json.loads(path.read_text(encoding="utf-8"))
    seen_ids: dict[str, int] = {}

    for rec in records:
        key = (ver, rec["page_number"])
        if key in MANUAL_OVERRIDES:
            rec["sheet_id"], rec["sheet_title"] = MANUAL_OVERRIDES[key]
        elif rec.get("verified_sheet_id"):
            rec["sheet_id"] = rec["verified_sheet_id"].strip().replace(" ", "")

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

    path.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"[{ver}] Synced {len(records)} sheets cleanly.")