"""
Step 5: Generate a grounded answer with citations.

Runs the full RAG pipeline for a sample query: retrieve context, build prompt,
call the LLM with temperature=0, and print the answer with source citations.

The LLM is instructed to answer only from retrieved context and to cite the
source filename after each relevant sentence. If the context is insufficient,
it should say so rather than hallucinate.

Usage:
    python step5_generate.py
"""

import weave
from dotenv import load_dotenv

from rag_pipeline import generate_answer

load_dotenv()

WANDB_PROJECT = "rag-tutorial"

SAMPLE_QUERY = "Can I upload screenshots, and how long are logs kept?"


def main() -> None:
    weave.init(WANDB_PROJECT)

    print("=" * 50)
    print("Step 5: Generate grounded answer")
    print("=" * 50)
    print(f"\nQuery: {SAMPLE_QUERY!r}\n")

    result = generate_answer(SAMPLE_QUERY, top_k=4)

    print("Answer")
    print("-" * 60)
    print(result["answer"])
    print("-" * 60)

    print(f"\nSources retrieved : {result['sources_retrieved']}")
    print(
        f"Token usage       : "
        f"{result['token_usage']['prompt_tokens']} prompt + "
        f"{result['token_usage']['completion_tokens']} completion"
    )
    print(
        "\nExpect to see citations like [image_policy.md] and "
        "[data_retention.md] in the answer. If citations are missing, "
        "check the prompt in rag_pipeline.build_prompt()."
    )
    print("\nDone. Run step6_evaluate.py next.")


if __name__ == "__main__":
    main()
