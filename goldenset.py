import json
import os
import random
from pathlib import Path
from pydantic import BaseModel, Field
from google import genai
from dotenv import load_dotenv

load_dotenv()
from google.genai import types


class GoldenItem(BaseModel):
    category: str = Field(description="One of: technical_lookup, version_diff, unchanged_check, set_level")
    question: str = Field(description="Realistic question an architect, contractor, or estimator would ask.")
    ground_truth: str = Field(description="Exact, factual answer strictly grounded in the provided sheet data.")
    expected_sheet_ids: list[str] = Field(description="List of Sheet IDs required to answer the question.")


class GoldenBatch(BaseModel):
    items: list[GoldenItem]


def build_draft_golden_set(output_path: str = "blueprint_data/golden_set_draft.json"):
    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"].strip(),
        http_options={"api_version": "v1beta"},
    )

    v2_sheets = json.loads(Path("blueprint_data/v2/enriched_manifest.json").read_text(encoding="utf-8"))
    diff_summary = json.loads(Path("blueprint_data/v2/diff_summary_v1_to_v2.json").read_text(encoding="utf-8"))

    # Sample diverse sheets across disciplines (modified + unchanged)
    modified = [s for s in v2_sheets if (s.get("diff_from_prev") or {}).get("status") == "MODIFIED"]
    unchanged = [s for s in v2_sheets if (s.get("diff_from_prev") or {}).get("status") in ("UNCHANGED", "UNCHANGED_MINOR")]

    random.seed(42)
    sample_sheets = random.sample(modified, min(10, len(modified))) + random.sample(unchanged, min(4, len(unchanged)))

    sheet_payloads = []
    for s in sample_sheets:
        diff = s.get("diff_from_prev") or {}
        sheet_payloads.append({
            "sheet_id": s["sheet_id"],
            "sheet_title": s["sheet_title"],
            "discipline": s.get("discipline"),
            "spatial_summary": s.get("spatial_summary"),
            "key_entities": s.get("key_entities", [])[:15],
            "referenced_sheets": s.get("referenced_sheets", []),
            "diff_status": diff.get("status"),
            "change_summary": diff.get("change_summary"),
            "cad_snippet": s.get("raw_cad_text", "")[:1500],
        })

    prompt = f"""
You are creating an evaluation benchmark (golden set) for a construction blueprint RAG system.
Generate 20 diverse, realistic engineering Q&A pairs from the sheet records and global diff summary below:
- 8 'technical_lookup' questions asking about specific specs, schedules, equipment, or room details in v2.
- 7 'version_diff' questions asking what changed between v1 and v2 on specific MODIFIED sheets.
- 3 'unchanged_check' questions asking if engineering changes occurred on UNCHANGED / UNCHANGED_MINOR sheets (ground truth must confirm no meaningful engineering changes occurred).
- 2 'set_level' questions about deleted sheets or overall v1-to-v2 revision scope.

Rules:
1. Every question MUST be answerable solely from the provided data.
2. Include exact Sheet IDs in `expected_sheet_ids` and mention them in `ground_truth`.
3. Never invent dimensions or specs not explicitly present in the text.

Global v1->v2 Summary:
Deleted sheets: {json.dumps(diff_summary.get("deleted_sheets", []))}

Sampled Sheet Records:
{json.dumps(sheet_payloads, indent=2)}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=GoldenBatch,
            temperature=0.2,
        ),
    )

    data = json.loads(response.text)
    Path(output_path).write_text(json.dumps(data["items"], indent=2), encoding="utf-8")
    print(f"Generated {len(data['items'])} draft golden samples -> {output_path}")


if __name__ == "__main__":
    build_draft_golden_set()