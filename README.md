# Multimodal Version-Aware Blueprint RAG

A production-oriented Retrieval-Augmented Generation (RAG) and automated revision-diffing pipeline built to ingest, compare, and query multi-version architectural and construction blueprint sets (`v1`: 154 pages, `v2`: 152 pages).

Unlike standard text-based RAG pipelines that fail on spatial drawings, vectorized CAD fonts, and exact sheet callouts, this system decouples **offline multimodal enrichment & OpenCV visual diffing** from **online hybrid retrieval (ChromaDB + BM25 via Reciprocal Rank Fusion)**.

---

## Key Engineering Highlights

* **VLM-Grounded Title Block Parsing:** Eliminates regex failures on unordered CAD text streams and vectorized fonts by using async `gemini-2.5-flash` vision extraction to verify canonical sheet IDs (`A152`, `S102`, `E801`) and resolve multi-sheet sets (`_DUP2`) without primary-key collisions.
* **Automated Visual Revision Diffing (`v1` $\rightarrow$ `v2`):** Combines OpenCV pixel-level red/green overlay generation with VLM semantic filtering to isolate real geometry and layout edits (e.g., room/equipment revisions on `A152`) while filtering out false-positive title-block stamp shifts (`UNCHANGED_MINOR`).
* **Hybrid Multi-Version Retrieval:** Fuses dense semantic search (`gemini-embedding-001`, 768-dim in ChromaDB) with lexical BM25 keyword matching via Reciprocal Rank Fusion (RRF) and exact Sheet ID boosting.
* **Intent-Aware Visual Grounding:** Automatically detects version-comparison queries (`diff_mode`) and passes both the high-resolution `v2` blueprint PNG and the red/green visual diff overlay directly into `gemini-2.5-flash` for citation-backed answers.

---

## System Architecture

```text
OFFLINE PREPROCESSING PIPELINE
───────────────────────────────────────────────────────────────────────────────
[v1.pdf (154p)] ──┐
                  ├──► 1_ingest_pdfs.py ──► High-DPI PNGs + Raw CAD Text
[v2.pdf (152p)] ──┘           │
                              ▼
                     2_enrich_vlm.py    ──► Async Gemini 2.5 Flash (Structured JSON:
                              │             verified_sheet_id, discipline, entities, summary)
                              ▼
                     3_diff_versions.py ──► OpenCV Red/Green Diff Overlays + VLM Change Filter
                              │             (MODIFIED / UNCHANGED_MINOR / DELETED)
                              ▼
                     4_index_chroma.py  ──► ChromaDB (gemini-embedding-001) + BM25 Corpus

ONLINE QUERY & MULTIMODAL GENERATION
───────────────────────────────────────────────────────────────────────────────
User Query ──► Intent Router (v1 | v2 | diff_mode)
                  │
                  ├──► ChromaDB Vector Search ──┐
                  └──► BM25 Keyword Search ─────┴──► Reciprocal Rank Fusion (RRF)
                                                               │
                                                               ▼
                                          Top-K Blueprint PNGs + Diff Overlays + CAD Context
                                                               │
                                                               ▼
                                                    Gemini 2.5 Flash (Grounded Answer + Citations)

flowchart TB
    subgraph OFFLINE["⚙️ OFFLINE PREPROCESSING & VERSION DIFFING"]
        direction TB
        PDFs["📄 Blueprint PDFs<br/>v1 (154p) & v2 (152p)"] --> Ingest["1_ingest_pdfs.py<br/>PyMuPDF 200-DPI PNGs + Raw CAD Text"]
        Ingest --> Enrich["2_enrich_vlm.py<br/>Async Gemini 2.5 Flash Vision<br/>(Verified Sheet ID, Discipline, Entities)"]
        Enrich --> Diff["3_diff_versions.py<br/>OpenCV Red/Green Pixel Diff + VLM Filter<br/>(MODIFIED / UNCHANGED_MINOR / DELETED)"]
        Diff --> Index["4_index_chroma.py<br/>Dual Indexing"]
        Index --> Chroma[("ChromaDB<br/>gemini-embedding-001 (768d)")]
        Index --> BM25[("BM25 Lexical Index<br/>Exact Sheet & Panel Tags")]
    end

    subgraph ONLINE["⚡ ONLINE HYBRID RETRIEVAL & MULTIMODAL QA"]
        direction TB
        Query["💬 User Query"] --> Router{"Intent Router<br/>v1 | v2 | diff_mode"}
        Router --> Chroma
        Router --> BM25
        Chroma --> RRF["Reciprocal Rank Fusion (RRF)<br/>+ Exact Sheet ID Boost"]
        BM25 --> RRF
        RRF --> Grounding["Top-K Blueprint PNGs<br/>+ Red/Green Diff Overlays + CAD Context"]
        Grounding --> VLM["Gemini 2.5 Flash<br/>Grounded Answer + Sheet Citations"]
    end

    style OFFLINE fill:#f8fafc,stroke:#94a3b8,stroke-width:1.5px
    style ONLINE fill:#f0fdf4,stroke:#22c55e,stroke-width:1.5px
    style VLM fill:#dbeafe,stroke:#2563eb,stroke-width:2px
