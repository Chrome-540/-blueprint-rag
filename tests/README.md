# Step tests

Each step has a folder. **Each change is its own file** (`v1_baseline.py`, `v2_...py`). The main file in the project
folder only changes when a version is approved.

| Step | Version | What changed | Status |
|---|---|---|---|
| 1_ingest | v1_baseline | The original `1_ingest_pdf.py` | Replaced by v3 |
| 1_ingest | v2_speed | Pages split across 6 processes, safe image saving. Same guessing as v1 | Run on v1.pdf: 154 pages in 25.9 s, 38 without an ID |
| 1_ingest | v3_position | v2 + sheet ID, title, date, revisions read by position | **In the app** (approved 2026-11-23). v1.pdf: 154 pages in 27.9 s, 3 without an ID, 0 wrong IDs |
| 2_enrich | v1_baseline | The original `2_enrich_async.py` | Replaced by v3 |
| 2_enrich | v2_highres | gemini-3.5-flash-lite, high resolution image, up to 30,000 characters of text, stricter prompt without project-specific examples, an ID must be printed in the PDF text | Run on 5 pages: good. References included the sheet itself, names on E801 could not be confirmed |
| 2_enrich | v3_checked | v2 + references found by code, names split into confirmed / unconfirmed, duplicates removed, image tokens recorded | **In the app** (approved 2026-11-23). 5 pages: good. Picture tables (E801) are not read reliably; high resolution has no effect on this model |
| 3_diff | v1_baseline | The original `3_diff_versions.py` (Gemini describes the changes) | Replaced by v2. Kept for reference, run.py does not run it |
| 3_diff | v2_no_ai | No Gemini: text differences from the PDF + red/green picture. Needs only ingest | **In the app** (approved 2026-11-23). 84 s. Known: 6 false alarms from a shifted stamp |
| 4_index | v1_baseline | The main `4_index_chroma.py` | In the app |
| 5_query | v1_baseline | The main `5_query_rag.py` | In the app |
| 6_router | v1_baseline | The original `backend/router.py` | Replaced by v2 |
| 6_router | v2_differences | New choice: show_differences | In the app, not run yet |

`1_ingest/_pipeline_parallel.py` is the `pipeline.py` that v2 and v3 need (now the main `pipeline.py`).
