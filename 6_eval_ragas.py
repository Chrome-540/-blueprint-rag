import json
import os
from importlib.machinery import SourceFileLoader
from pathlib import Path

from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    MultiModalFaithfulness,
)
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.metrics.collections import MultiModalFaithfulness
import ragas.prompt.multi_modal_prompt as image_policy
import os
from dotenv import load_dotenv

load_dotenv()
rag_module = SourceFileLoader("query_rag", "5_query_rag.py").load_module()


def run_golden_eval(golden_path: str = "blueprint_data\\golden_set_draft.json", top_k: int = 3):
    api_key = os.environ["GEMINI_API_KEY"].strip()
    golden_samples = json.loads(Path(golden_path).read_text(encoding="utf-8"))

    # Allow Ragas multimodal judge to inspect local blueprint & diff PNGs
    image_policy.ALLOW_LOCAL_FILE_ACCESS = True
    image_policy.ALLOWED_IMAGE_BASE_DIR = str(Path("blueprint_data").resolve())

    judge_llm = LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key,
            temperature=0.0,
        ),
        bypass_n=True,
    )
    judge_embeddings = LangchainEmbeddingsWrapper(
        GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key,
        )
    )

    rag = rag_module.BlueprintRAG()
    rows = []
    retrieval_hits = 0

    print(f"Evaluating {len(golden_samples)} golden samples (top_k={top_k})...")

    for idx, sample in enumerate(golden_samples, start=1):
        q = sample["question"]
        expected_ids = set(sample.get("expected_sheet_ids", []))

        version, is_diff = rag._detect_query_mode(q)
        hits = rag.hybrid_search(q, version=version, top_k=top_k)
        retrieved_ids = {h["metadata"]["sheet_id"] for h in hits}

        # Deterministic Retriever Hit-Rate check
        hit_ok = expected_ids.issubset(retrieved_ids) if expected_ids else True
        if hit_ok:
            retrieval_hits += 1

        # Build contexts matching exact text + images seen by rag.ask()
        retrieved_contexts = []
        if is_diff:
            summary_file = Path("blueprint_data/v2/diff_summary_v1_to_v2.json")
            if summary_file.exists():
                summary_data = json.loads(summary_file.read_text(encoding="utf-8"))
                deleted = summary_data.get("deleted_sheets", [])
                modified = [m["sheet_id"] for m in summary_data.get("modified_sheets", [])]
                retrieved_contexts.append(
                    f"Global v1 -> v2 Summary: {len(modified)} sheets modified ({', '.join(modified[:30])}...), "
                    f"Deleted from v1: {json.dumps(deleted)}."
                )

        for h in hits:
            meta = h["metadata"]
            retrieved_contexts.append(
                f"Sheet {meta['sheet_id']} ({meta['version']}, Page {meta['page_number']}): "
                f"{meta['change_summary']}\n{h['text'][:3500]}"
            )
            img_path = Path(meta["image_path"])
            if img_path.exists():
                retrieved_contexts.append(str(img_path.resolve()))

            overlay_str = meta.get("diff_overlay_path")
            if is_diff and overlay_str and Path(overlay_str).exists():
                retrieved_contexts.append(str(Path(overlay_str).resolve()))

        generated_answer = rag.ask(q, top_k=top_k)
        print(f"  [{idx}/{len(golden_samples)}] Hit={hit_ok} | Expected={list(expected_ids)} | Got={list(retrieved_ids)}")

        rows.append(
            {
                "user_input": q,
                "retrieved_contexts": retrieved_contexts,
                "response": generated_answer,
                "reference": sample["ground_truth"],
            }
        )

    hit_rate = retrieval_hits / max(len(golden_samples), 1)
    print(f"\nDeterministic Retriever Hit-Rate (Recall@{top_k}): {hit_rate:.2%} ({retrieval_hits}/{len(golden_samples)})")

    eval_dataset = Dataset.from_list(rows)
    print("\nRunning Ragas Multimodal & Retrieval Evaluation...")
    result = evaluate(
        dataset=eval_dataset,
        metrics=[
            MultiModalFaithfulness(llm=judge_llm, embeddings=judge_embeddings),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    print("\n=== GOLDEN SET RAGAS SCORES ===")
    print(result)

    df = result.to_pandas()
    df["category"] = [s.get("category", "uncategorized") for s in golden_samples]
    df["expected_sheets"] = [",".join(s.get("expected_sheet_ids", [])) for s in golden_samples]
    out_csv = "blueprint_data/golden_eval_results.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nPer-question breakdown saved -> {out_csv}")


if __name__ == "__main__":
    run_golden_eval()