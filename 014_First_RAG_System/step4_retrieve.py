"""
Step 4: Retrieve evidence for a user question.

Loads the FAISS index built in step 3, embeds a sample query,
and prints the top-k results with similarity scores and source previews.

This shows what context the LLM will receive before any generation happens.
Inspecting retrieval output separately makes it easy to catch problems
(wrong sources, low scores, broken chunks) before blaming the generator.

Usage:
    python step4_retrieve.py
"""

import weave
from dotenv import load_dotenv

from rag_pipeline import retrieve_chunks

load_dotenv()

WANDB_PROJECT = "rag-tutorial"

# A query that requires evidence from two different documents
SAMPLE_QUERY = "Can I upload screenshots, and how long are logs kept?"


def print_results(results: list[dict]) -> None:
    col_rank = 5
    col_score = 8
    col_src = 28
    header = (
        f"{'Rank':<{col_rank}}  {'Score':>{col_score}}  "
        f"{'Source':<{col_src}}  Preview"
    )
    print(header)
    print("-" * (len(header) + 30))
    for r in results:
        preview = r["text"].replace("\n", " ")[:70].strip()
        print(
            f"{r['rank']:<{col_rank}}  "
            f"{r['score']:>{col_score}.4f}  "
            f"{r['source']:<{col_src}}  "
            f"{preview}..."
        )


def main() -> None:
    weave.init(WANDB_PROJECT)

    print("=" * 50)
    print("Step 4: Retrieve evidence")
    print("=" * 50)
    print(f"\nQuery: {SAMPLE_QUERY!r}\n")

    results = retrieve_chunks(SAMPLE_QUERY, top_k=4)

    # Add rank field for display
    for i, r in enumerate(results, 1):
        r["rank"] = i

    print_results(results)
    print(
        "\nCheck that the top results match the expected sources "
        "(image_policy.md, data_retention.md). "
        "If they don't, the likely culprit is chunk quality — review step 2."
    )
    print("\nDone. Run step5_generate.py next.")


if __name__ == "__main__":
    main()
