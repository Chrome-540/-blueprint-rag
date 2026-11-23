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
| 4_index | v1_baseline | The original `4_index_chroma.py` | Replaced by v2 |
| 4_index | v2_full_text | Stores and keyword-searches the full text of every sheet (was: first 8,000 characters) | **In the app** (approved 2026-11-23) |
| 4_index | v3_clean | v2 + repeated notes left out of the vector, retries and timeout, old entries of a file deleted first, unconfirmed names for search only | Search not better than v2 (75 vs 76 of 95). Not approved |
| 5_query | v1_baseline | The original `5_query_rag.py` | Replaced by v3 |
| 5_query | v4_search_extra | v3 + keyword search also uses the unconfirmed names. Needs 4_index v3_clean | Not run yet |
| 5_query | v5_fusion | v4 + the #1 of keyword and of meaning search are always kept, text of 5 sheets, images of 3 | Search 93% (v3: 80%), answers 11 of 12, 17.6 s per question. **In the app** (approved 2026-11-23) |
| 6_router | v1_baseline | The original `backend/router.py` | Replaced by v2 |
| 5_query | v2_model | Same model as Enrich (gemini-3.5-flash-lite) | 3 questions: good |
| 5_query | v3_full_text | v2 + Gemini gets the full text of the sheets found (was: first 4,000 characters). Needs 4_index v2_full_text | Replaced by v5. 1 question right, 13.2 s. Answers are not checked by code |
| 6_router | v2_differences | New choice: show_differences | Replaced by v3 |
| 6_router | v3_model | v2 with the same model as Enrich, without the thinking setting it rejects | **In the app** (approved 2026-11-23). 10 of 10 right |

`1_ingest/_pipeline_parallel.py` is the `pipeline.py` that v2 and v3 need (now the main `pipeline.py`).
