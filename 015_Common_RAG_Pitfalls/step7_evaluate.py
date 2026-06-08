"""
Step 7: Evaluate the RAG pipeline with W&B Weave.

Runs a weave.Evaluation over six examples using three scorers:
  source_hit_scorer  -- did the retriever find the expected source?
  keyword_scorer     -- does the answer contain the expected keywords?
  citation_scorer    -- did the model include [source: ...] citations?

After this step, open the Evaluations tab in your Weights & Biases project
to see scores broken down per example and per scorer.

Usage:
    python step7_evaluate.py

Requires:
    OPENAI_API_KEY in .env
    chroma_db/ created by step3_embed_index.py
"""

import asyncio
import os
import re
import sys

import weave
from dotenv import load_dotenv

from rag_pipeline import RETRIEVAL_TOP_K, WANDB_PROJECT, rag_query

load_dotenv()

# ─── Evaluation dataset ───────────────────────────────────────────────────────

EVAL_DATASET = [
    {
        "id":               "q1",
        "question":         "What image formats does the platform support?",
        "expected_sources": ["image_policy.md"],
        "required_keywords": ["PNG", "JPEG", "WebP"],
    },
    {
        "id":               "q2",
        "question":         "How long does the platform keep API request logs?",
        "expected_sources": ["data_retention.md"],
        "required_keywords": ["30 days"],
    },
    {
        "id":               "q3",
        "question":         "What is the minimum number of training examples for fine-tuning?",
        "expected_sources": ["custom_llm_guide.md"],
        "required_keywords": ["50"],
    },
    {
        "id":               "q4",
        "question":         "What does a 429 error mean and how should I handle it?",
        "expected_sources": ["support_faq.md"],
        "required_keywords": ["rate limit", "backoff"],
    },
    {
        "id":               "q5",
        "question":         "Which embedding model is recommended for new projects?",
        "expected_sources": ["api_models.md"],
        "required_keywords": ["text-embedding-3-small"],
    },
    {
        "id":               "q6",
        "question":         "Can I upload screenshots, and how long are logs kept?",
        "expected_sources": ["image_policy.md", "data_retention.md"],
        "required_keywords": ["PNG", "30 days"],
    },
]

# ─── Model ────────────────────────────────────────────────────────────────────

class RAGModel(weave.Model):
    """Wraps rag_query() as a weave.Model so weave.Evaluation can call it."""
    top_k: int = RETRIEVAL_TOP_K

    @weave.op()
    def predict(self, question: str) -> dict:
        return rag_query(question, top_k=self.top_k)

# ─── Scorers ──────────────────────────────────────────────────────────────────

@weave.op()
def source_hit_scorer(output: dict, expected_sources: list) -> dict:
    """
    Check whether every expected source appeared in the retrieved docs.

    A miss means the retriever failed, not the generator. This score lets
    you separate retrieval problems from generation problems cleanly.
    """
    retrieved = [d["source"] for d in output.get("retrieved_docs", [])]
    hits  = [s for s in expected_sources if s in retrieved]
    score = len(hits) / len(expected_sources) if expected_sources else 0.0
    return {
        "source_hit": score,
        "hits":       hits,
        "missed":     list(set(expected_sources) - set(hits)),
    }


@weave.op()
def keyword_scorer(output: dict, required_keywords: list) -> dict:
    """
    Check whether the answer contains every required keyword (case-insensitive).

    A low score typically means the model retrieved the right source but
    generated a vague answer, or a weak retrieval chunk was returned.
    """
    answer = output.get("answer", "").lower()
    found  = [kw for kw in required_keywords if kw.lower() in answer]
    score  = len(found) / len(required_keywords) if required_keywords else 0.0
    return {
        "keyword_score": score,
        "found":         found,
        "missing":       list(set(required_keywords) - set(found)),
    }


@weave.op()
def citation_scorer(output: dict, expected_sources: list) -> dict:
    """
    Check whether the answer contains [source: ...] citation markers.

    This catches the Pitfall 4 failure mode: a correct-looking answer with
    no citations that is impossible to audit or verify.
    """
    answer    = output.get("answer", "")
    citations = re.findall(r"\[source:\s*([^\]]+)\]", answer, re.IGNORECASE)
    return {
        "has_citation":    len(citations) > 0,
        "citations_found": citations,
    }

# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("\n=== Step 7: Evaluate with W&B Weave ===\n")

    if not os.path.exists("chroma_db"):
        print("ERROR: chroma_db/ not found. Run step3_embed_index.py first.")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Check your .env file.")
        sys.exit(1)

    weave.init(WANDB_PROJECT)

    model = RAGModel()
    evaluation = weave.Evaluation(
        evaluation_name="rag-baseline-eval",
        dataset=EVAL_DATASET,
        scorers=[source_hit_scorer, keyword_scorer, citation_scorer],
    )

    print(f"Running {len(EVAL_DATASET)} evaluation examples...\n")
    results = asyncio.run(evaluation.evaluate(model))

    print("\n=== Aggregate scores ===")
    print(results)
    print()
    print("Open your Weights & Biases project, click 'Evaluations', and select")
    print("'rag-baseline-eval' to see per-example scores, per-scorer breakdowns,")
    print("and what each retrieval call returned.")
    print("\nDone. Open the W&B Weave dashboard (Step 8) to inspect traces.")


if __name__ == "__main__":
    main()
