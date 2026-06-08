"""
Step 6: Evaluate the RAG system with W&B Weave.

Runs a six-question evaluation dataset through the full RAG pipeline.
Three scorers measure:
  - source_hit_rate   : fraction of expected source docs that were retrieved
  - keyword_coverage  : fraction of required keywords present in the answer
  - has_citation      : 1 if the answer contains at least one [filename.md] citation

All calls are traced in W&B Weave. After the run, open your Weights & Biases
project to see the eval dashboard with per-example scores and full trace trees.

Usage:
    python step6_evaluate.py
"""

import asyncio
import re

import weave
from dotenv import load_dotenv

from rag_pipeline import generate_answer

load_dotenv()

WANDB_PROJECT = "rag-tutorial"


# ─── Weave Model ──────────────────────────────────────────────────────────────


class RAGModel(weave.Model):
    """
    Wraps the RAG pipeline for Weave evaluation.
    Attributes become versioned model parameters visible in the W&B UI.
    """

    top_k: int = 4
    generation_model: str = "gpt-4o-mini"

    @weave.op()
    def predict(self, question: str) -> dict:
        return generate_answer(question, top_k=self.top_k)


# ─── Scorers ──────────────────────────────────────────────────────────────────
# Scorer parameter names must match dataset column names exactly.
# The `output` parameter always receives the return value of predict().


@weave.op()
def source_hit_scorer(output: dict, expected_sources: list) -> dict:
    """
    Measures what fraction of the expected source documents were retrieved.
    A score of 1.0 means every expected document appeared in the context.
    """
    retrieved = set(output.get("sources_retrieved", []))
    expected = set(expected_sources)
    if not expected:
        return {"source_hit_rate": 0.0}
    hits = len(expected & retrieved)
    return {"source_hit_rate": round(hits / len(expected), 2)}


@weave.op()
def keyword_scorer(output: dict, required_keywords: list) -> dict:
    """
    Measures what fraction of required keywords appear in the answer (case-insensitive).
    Useful as a quick proxy for factual completeness.
    """
    answer = output.get("answer", "").lower()
    if not required_keywords:
        return {"keyword_coverage": 0.0}
    hits = sum(1 for kw in required_keywords if kw.lower() in answer)
    return {"keyword_coverage": round(hits / len(required_keywords), 2)}


@weave.op()
def citation_scorer(output: dict) -> dict:
    """
    Checks whether the answer contains at least one citation in [filename.md] format.
    Returns 1 if citations are present, 0 if not.
    """
    answer = output.get("answer", "")
    # Match patterns like [image_policy.md] or [support_faq.md]
    has_citation = bool(re.search(r"\[\w[\w_\-]*\.\w+\]", answer))
    return {"has_citation": int(has_citation)}


# ─── Evaluation dataset ───────────────────────────────────────────────────────

EVAL_DATASET = [
    {
        "id": "q1",
        "question": "Can I upload screenshots to use with the API?",
        "expected_sources": ["image_policy.md"],
        "required_keywords": ["png", "jpeg", "upload"],
    },
    {
        "id": "q2",
        "question": "How long does the platform keep my API request logs?",
        "expected_sources": ["data_retention.md"],
        "required_keywords": ["30 days"],
    },
    {
        "id": "q3",
        "question": "What is the minimum number of training examples needed to fine-tune a custom LLM?",
        "expected_sources": ["custom_llm_guide.md"],
        "required_keywords": ["50"],
    },
    {
        "id": "q4",
        "question": "What does a 429 error mean and how should I handle it?",
        "expected_sources": ["support_faq.md"],
        "required_keywords": ["rate limit", "backoff"],
    },
    {
        "id": "q5",
        "question": "Which embedding model should I use for a new project?",
        "expected_sources": ["api_models.md"],
        "required_keywords": ["text-embedding-3-small"],
    },
    {
        "id": "q6",
        "question": "Can I upload screenshots, and how long are logs kept?",
        "expected_sources": ["image_policy.md", "data_retention.md"],
        "required_keywords": ["png", "30 days"],
    },
]


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    weave.init(WANDB_PROJECT)

    print("=" * 50)
    print("Step 6: Evaluate RAG system")
    print("=" * 50)
    print(f"\nRunning {len(EVAL_DATASET)} evaluation examples...\n")

    model = RAGModel()

    evaluation = weave.Evaluation(
        evaluation_name="rag-baseline-eval",
        dataset=EVAL_DATASET,
        scorers=[source_hit_scorer, keyword_scorer, citation_scorer],
    )

    results = asyncio.run(evaluation.evaluate(model))

    print("\n=== Aggregate scores ===")
    print(results)
    print(
        "\nOpen your Weights & Biases project to see the full eval dashboard "
        "with per-example scores, trace trees, and retrieved context."
    )
    print(
        "URL format: https://wandb.ai/<your-entity>/rag-tutorial/weave/evaluations"
    )


if __name__ == "__main__":
    main()
