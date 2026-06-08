"""
Step 4: Retrieval smoke test — verify retrieval before adding the LLM.

Test the retriever against four queries with known expected sources.
This step runs no generation. If retrieval fails here, no prompt or LLM
can fix it.

Usage:
    python step4_smoke_test.py

Requires:
    OPENAI_API_KEY in .env
    chroma_db/ created by step3_embed_index.py
"""

import os
import sys

import weave
from dotenv import load_dotenv

from rag_pipeline import CHROMA_DIR, RETRIEVAL_TOP_K, WANDB_PROJECT, retrieve_docs

load_dotenv()

TEST_QUERIES = [
    {
        "query":           "What image formats does the platform support?",
        "expected_source": "image_policy.md",
    },
    {
        "query":           "What is the minimum number of training examples for fine-tuning?",
        "expected_source": "custom_llm_guide.md",
    },
    {
        "query":           "What does a 429 error mean and how should I handle it?",
        "expected_source": "support_faq.md",
    },
    {
        "query":           "Which embedding model is recommended for new projects?",
        "expected_source": "api_models.md",
    },
]


def run_smoke_test(query: str, expected_source: str) -> bool:
    """Retrieve chunks, print results, return True if expected source is found."""
    print(f"  Query: {query}")
    docs = retrieve_docs(query, top_k=RETRIEVAL_TOP_K)

    sources_found = [d["source"] for d in docs]
    hit = expected_source in sources_found

    for d in docs:
        marker = "✓" if d["source"] == expected_source else " "
        preview = d["page_content"][:90].replace("\n", " ")
        print(f"    [{marker}] score={d['score']:.4f}  {d['source']}  [{d['chunk_id']}]")
        print(f"        {preview}...")

    if hit:
        print(f"  PASS: {expected_source} found in top {RETRIEVAL_TOP_K}\n")
    else:
        print(f"  FAIL: {expected_source} not in top {RETRIEVAL_TOP_K}")
        print("  Diagnostic: try a more specific query, check chunk sizes,")
        print("  or verify the index was built with the correct embedding model.\n")
    return hit


def main() -> None:
    print("\n=== Step 4: Retrieval smoke test ===\n")

    if not os.path.exists(CHROMA_DIR):
        print(f"ERROR: {CHROMA_DIR}/ not found. Run step3_embed_index.py first.")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Check your .env file.")
        sys.exit(1)

    weave.init(WANDB_PROJECT)

    passed = 0
    for item in TEST_QUERIES:
        ok = run_smoke_test(item["query"], item["expected_source"])
        if ok:
            passed += 1

    total = len(TEST_QUERIES)
    print(f"Smoke test: {passed}/{total} queries found their expected source")

    if passed == total:
        print("\nRetrieval looks good. Run step5_rag_chain.py next.")
    else:
        print("\nSome queries missed their expected source.")
        print("Fix retrieval before connecting the LLM.\n")
        print("Common causes:")
        print("  - chunk_size too large (whole-document chunks dilute context)")
        print("  - no overlap (key facts split across chunk boundaries)")
        print("  - query too vague (try more specific phrasing)")
        print("  - embedding model changed without rebuilding the index")


if __name__ == "__main__":
    main()
