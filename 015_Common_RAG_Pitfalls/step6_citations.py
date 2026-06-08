"""
Step 6: Add source citations and refusal behavior.

Demonstrates the difference between a weak prompt and a citation-enforcing
prompt using the same query and the same retrieved context.

The weak prompt ("You are a helpful assistant. Answer the user's question.")
allows the model to fill gaps from training data without citing anything.
The strong prompt requires [source: filename] citations after every factual
claim and instructs the model to refuse questions not covered by the context.

Usage:
    python step6_citations.py

Requires:
    OPENAI_API_KEY in .env
    chroma_db/ created by step3_embed_index.py
"""

import os
import re
import sys

import weave
from dotenv import load_dotenv

from rag_pipeline import (
    CHROMA_DIR,
    RETRIEVAL_TOP_K,
    WANDB_PROJECT,
    WEAK_SYSTEM_PROMPT,
    generate_answer,
    retrieve_docs,
)

load_dotenv()

# Two queries: one that reveals hallucination, one that tests refusal
QUERIES = [
    "What is the maximum image file size, and how long are images stored?",
    "What is the capital of France?",
]


def main() -> None:
    print("\n=== Step 6: Citation enforcement and refusal behavior ===\n")

    if not os.path.exists(CHROMA_DIR):
        print(f"ERROR: {CHROMA_DIR}/ not found. Run step3_embed_index.py first.")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Check your .env file.")
        sys.exit(1)

    weave.init(WANDB_PROJECT)

    for query in QUERIES:
        print(f"Query: {query}\n")
        docs = retrieve_docs(query, top_k=RETRIEVAL_TOP_K)

        # Bad: weak prompt
        result_bad = generate_answer(query, docs, system_prompt_template=WEAK_SYSTEM_PROMPT)
        bad_cites  = re.findall(r"\[source:[^\]]+\]", result_bad["answer"], re.IGNORECASE)
        print("  BAD  (weak prompt, no citation requirement):")
        print(f"    {result_bad['answer'][:500]}")
        print(f"    Citations found: {len(bad_cites)}")
        print()

        # Good: citation-enforcing prompt (default in rag_pipeline.py)
        result_good = generate_answer(query, docs)
        good_cites  = re.findall(r"\[source:[^\]]+\]", result_good["answer"], re.IGNORECASE)
        print("  GOOD (citation-enforcing prompt):")
        print(f"    {result_good['answer'][:500]}")
        print(f"    Citations found: {len(good_cites)}")
        print()
        print("-" * 70)
        print()

    print("What to look for:")
    print("  The bad answer may invent specific facts not present in the docs.")
    print("  The good answer cites [source: filename] for every factual claim.")
    print("  For out-of-scope questions both prompts should refuse, but only")
    print("  the strong prompt gives a consistent, auditable refusal phrase.")
    print("\nDone. Run step7_evaluate.py next.")


if __name__ == "__main__":
    main()
