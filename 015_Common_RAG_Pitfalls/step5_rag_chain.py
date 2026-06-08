"""
Step 5: Build the RAG chain.

Connects retrieval to generation through rag_query(). Runs five in-scope
questions (expect cited answers) and two out-of-scope questions (expect
explicit refusals). Every call is traced in W&B Weave.

Usage:
    python step5_rag_chain.py

Requires:
    OPENAI_API_KEY in .env
    chroma_db/ created by step3_embed_index.py
"""

import os
import sys

import weave
from dotenv import load_dotenv

from rag_pipeline import CHROMA_DIR, RETRIEVAL_TOP_K, WANDB_PROJECT, rag_query

load_dotenv()

TEST_QUESTIONS = [
    # In-scope — expect cited answers
    "What image formats does the platform support?",
    "How long does the platform keep API request logs?",
    "What is the minimum number of training examples for a fine-tuned model?",
    "What does a 429 error mean and how should I handle it?",
    "Which embedding model should I use for a new project?",
    # Out-of-scope — expect explicit refusal
    "What is the capital of France?",
    "How do I delete my account permanently?",
]


def print_result(result: dict) -> None:
    sources = [d["source"] for d in result["retrieved_docs"]]
    print(f"  Q: {result['query']}")
    print(f"  Sources retrieved: {sources}")
    print(f"  A: {result['answer']}")
    print()


def main() -> None:
    print("\n=== Step 5: RAG chain ===\n")

    if not os.path.exists(CHROMA_DIR):
        print(f"ERROR: {CHROMA_DIR}/ not found. Run step3_embed_index.py first.")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Check your .env file.")
        sys.exit(1)

    weave.init(WANDB_PROJECT)
    print("Running questions through the RAG chain...\n")
    print("-" * 70)

    print("In-scope questions (expect cited answers):\n")
    for q in TEST_QUESTIONS[:-2]:
        result = rag_query(q, top_k=RETRIEVAL_TOP_K)
        print_result(result)

    print("-" * 70)
    print("Out-of-scope questions (expect explicit refusal):\n")
    for q in TEST_QUESTIONS[-2:]:
        result = rag_query(q, top_k=RETRIEVAL_TOP_K)
        print_result(result)

    print("-" * 70)
    print("\nWhat to look for:")
    print("  In-scope answers should contain [source: filename] citations.")
    print("  Out-of-scope answers should contain the refusal phrase, not a")
    print("  confident wrong answer from training data.")
    print("\nIf in-scope answers are missing citations, see step6_citations.py")
    print("for a side-by-side comparison showing why the prompt matters.")
    print("\nDone. Run step6_citations.py next.")


if __name__ == "__main__":
    main()
