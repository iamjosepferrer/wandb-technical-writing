"""
Pitfalls demo: all five failure modes, before and after.

Demonstrates five failure modes that appear in real RAG pipelines.
Pitfalls 1 and 2 require no API calls (text processing only).
Pitfalls 3, 4, and 5 require the Chroma index and OpenAI API key.

Usage:
    python pitfalls_demo.py --all
    python pitfalls_demo.py 1
    python pitfalls_demo.py 1 2

Pitfalls:
  1. Chunks too large    -- retriever returns whole documents, not passages
  2. No chunk overlap    -- facts at boundaries disappear from retrieval
  3. No smoke test       -- LLM produces confident wrong answer, undetected
  4. Weak prompt         -- answers with no citations, impossible to audit
  5. Vague query design  -- retriever returns random-looking chunks

Requires:
  docs/       created by step1_knowledge_base.py
  chroma_db/  created by step3_embed_index.py  (pitfalls 3, 4, 5)
  OPENAI_API_KEY in .env                       (pitfalls 3, 4, 5)
"""

import os
import re
import sys

from dotenv import load_dotenv

load_dotenv()


def separator(title: str) -> None:
    bar = "-" * 68
    print(f"\n{bar}")
    print(f"  {title}")
    print(f"{bar}")


def require_index() -> bool:
    if os.path.exists("chroma_db"):
        return True
    print("\n  ERROR: chroma_db/ not found. Run step3_embed_index.py first.")
    return False


def require_api_key() -> bool:
    if os.environ.get("OPENAI_API_KEY"):
        return True
    print("\n  ERROR: OPENAI_API_KEY not set. Check your .env file.")
    return False


# ─── Pitfall 1: Chunks too large ──────────────────────────────────────────────

def pitfall_chunk_size() -> None:
    separator("Pitfall 1: Chunks too large")
    print("""
Problem:
  With chunk_size=8000, every document fits in a single chunk. The retriever
  returns entire documents instead of targeted passages. The LLM receives
  the full content of every document alongside the one sentence it needs.
  This dilutes the signal and raises the chance the model ignores the
  correct information buried in the noise.
""")
    from rag_pipeline import DATA_DIR, chunk_documents, load_documents

    if not os.path.isdir(DATA_DIR):
        print(f"  ERROR: {DATA_DIR}/ not found. Run step1_knowledge_base.py first.")
        return

    docs = load_documents(DATA_DIR)

    configs = [
        ("BAD  chunk_size=8000, overlap=0",     8000, 0),
        ("GOOD chunk_size=1600, overlap=320",   1600, 320),
    ]

    for label, cs, ov in configs:
        chunks = chunk_documents(docs, chunk_size=cs, chunk_overlap=ov)
        avg    = sum(c.metadata["char_count"] for c in chunks) / len(chunks)
        print(f"  {label}")
        print(f"    chunks produced : {len(chunks)}")
        print(f"    avg chunk chars : {avg:.0f}")
        if label.startswith("BAD"):
            sample = chunks[0].page_content[:300].replace("\n", " ")
            print(f"    first 300 chars : {sample}...")
        print()

    print("""Fix:
  Use chunk_size=1600 with overlap=320 (the defaults in rag_pipeline.py).
  The retriever returns targeted passages; the LLM only sees the relevant
  section, not the entire document.
""")


# ─── Pitfall 2: No chunk overlap ──────────────────────────────────────────────

def pitfall_no_overlap() -> None:
    separator("Pitfall 2: No chunk overlap")
    print("""
Problem:
  With overlap=0, chunk boundaries are hard cuts. Any idea that spans two
  chunks is split in half. A query that targets the second half of a concept
  will retrieve a chunk that starts mid-thought, missing the setup.
""")
    from rag_pipeline import DATA_DIR, chunk_documents, load_documents

    if not os.path.isdir(DATA_DIR):
        print(f"  ERROR: {DATA_DIR}/ not found. Run step1_knowledge_base.py first.")
        return

    docs   = load_documents(DATA_DIR)
    target = next((d for d in docs if d.metadata["source"] == "image_policy.md"), docs[0])

    no_ov   = chunk_documents([target], chunk_size=1600, chunk_overlap=0)
    with_ov = chunk_documents([target], chunk_size=1600, chunk_overlap=320)

    if len(no_ov) >= 2:
        print("  BAD  overlap=0")
        print(f"    end of chunk 0 :   ...{no_ov[0].page_content[-130:].replace(chr(10), ' ').strip()}")
        print(f"    start of chunk 1:   {no_ov[1].page_content[:130].replace(chr(10), ' ').strip()}...")
        print()

    if len(with_ov) >= 2:
        print("  GOOD overlap=320")
        print(f"    end of chunk 0 :   ...{with_ov[0].page_content[-130:].replace(chr(10), ' ').strip()}")
        print(f"    start of chunk 1:   {with_ov[1].page_content[:130].replace(chr(10), ' ').strip()}...")
        print()

    print("""Fix:
  Set overlap to roughly 20 percent of chunk_size. At chunk_size=1600 that
  is 320 chars. The overlap carries context from the end of each chunk into
  the start of the next, so boundary-spanning facts remain retrievable from
  either side.
""")


# ─── Pitfall 3: No retrieval smoke test ───────────────────────────────────────

def pitfall_no_smoke_test() -> None:
    separator("Pitfall 3: Skipping the retrieval smoke test")
    print("""
Problem:
  If you add the LLM before testing retrieval, broken retrieval is invisible.
  The model generates a fluent, confident-sounding answer regardless of what
  the retriever returned, including completely wrong chunks.
""")
    if not require_index() or not require_api_key():
        return

    import weave
    from rag_pipeline import WANDB_PROJECT, generate_answer, retrieve_docs

    weave.init(WANDB_PROJECT)

    vague_query = "What are the policies?"
    print(f"  Query (vague): '{vague_query}'\n")

    docs = retrieve_docs(vague_query, top_k=4)
    print("  Retrieved chunks (low scores, mixed sources):")
    for d in docs:
        print(f"    score={d['score']:.4f}  {d['source']}")
    print()

    result = generate_answer(vague_query, docs)
    print("  Generated answer (may be generic or refuse due to low-relevance chunks):")
    print(f"    {result['answer'][:400]}...")
    print()

    specific_query = "What image formats does the platform support?"
    print(f"  Specific query: '{specific_query}'\n")
    docs2 = retrieve_docs(specific_query, top_k=4)
    print("  Retrieved chunks (higher scores, correct source at top):")
    for d in docs2:
        print(f"    score={d['score']:.4f}  {d['source']}")
    print()
    result2 = generate_answer(specific_query, docs2)
    print("  Generated answer (specific retrieval, cited answer):")
    print(f"    {result2['answer'][:400]}")

    print("""
Fix:
  Run step4_smoke_test.py before step5_rag_chain.py. Verify each expected
  source appears in the top results before connecting the LLM. Retrieval
  problems masked by the generator are the hardest RAG bugs to debug.
""")


# ─── Pitfall 4: Weak prompt, no citations ─────────────────────────────────────

def pitfall_weak_prompt() -> None:
    separator("Pitfall 4: Weak prompt, no citation enforcement")
    print("""
Problem:
  A vague instruction ("be helpful", "answer accurately") does not change
  model behaviour. Without an explicit citation requirement, the model
  produces smooth prose with no traceability to source documents.
  Without a refusal instruction, it fills knowledge gaps from training data.
""")
    if not require_index() or not require_api_key():
        return

    import weave
    from rag_pipeline import WANDB_PROJECT, WEAK_SYSTEM_PROMPT, generate_answer, retrieve_docs

    weave.init(WANDB_PROJECT)

    query = "What is the maximum image file size, and how long are images stored?"
    docs  = retrieve_docs(query, top_k=4)
    print(f"  Query: {query}\n")

    result_bad  = generate_answer(query, docs, system_prompt_template=WEAK_SYSTEM_PROMPT)
    result_good = generate_answer(query, docs)

    bad_cites  = re.findall(r"\[source:[^\]]+\]", result_bad["answer"],  re.IGNORECASE)
    good_cites = re.findall(r"\[source:[^\]]+\]", result_good["answer"], re.IGNORECASE)

    print("  BAD  (weak prompt, no citation requirement):")
    print(f"    {result_bad['answer'][:500]}")
    print(f"    Citations: {len(bad_cites)}")
    print()
    print("  GOOD (citation-enforcing prompt):")
    print(f"    {result_good['answer'][:500]}")
    print(f"    Citations: {len(good_cites)}")
    print("""
Fix:
  Use SYSTEM_PROMPT from rag_pipeline.py. It explicitly requires
  [source: filename] citations after every factual claim and instructs
  the model to refuse questions not covered by the context.
""")


# ─── Pitfall 5: Vague query design ───────────────────────────────────────────

def pitfall_vague_query() -> None:
    separator("Pitfall 5: Query too vague for retrieval to help")
    print("""
Problem:
  The embedding similarity between "What are the rules?" and any specific
  document section is low. The retriever returns the chunks that are least
  dissimilar, which may be correct or completely wrong. Vague queries
  produce low, noisy similarity scores that a good prompt cannot fix.
  The pipeline is working correctly. The problem is the query design.
""")
    if not require_index() or not require_api_key():
        return

    import weave
    from rag_pipeline import WANDB_PROJECT, retrieve_docs

    weave.init(WANDB_PROJECT)

    pairs = [
        ("VAGUE",    "What are the rules?"),
        ("SPECIFIC", "What image formats are supported and what is the maximum file size?"),
        ("VAGUE",    "How does it work?"),
        ("SPECIFIC", "How does exponential backoff work for a 429 rate limit error?"),
    ]

    for label, q in pairs:
        docs    = retrieve_docs(q, top_k=3)
        sources = [d["source"] for d in docs]
        scores  = [f"{d['score']:.4f}" for d in docs]
        print(f"  {label:<9} '{q}'")
        print(f"    sources : {sources}")
        print(f"    scores  : {scores}")
        print()

    print("""Fix:
  Write queries that use the same vocabulary as the documents.
  If users will send vague queries, add a query-rewriting step before
  retrieval: ask a small LLM to expand the vague query into a specific
  one using domain vocabulary, then embed the rewritten version.
""")


# ─── Runner ───────────────────────────────────────────────────────────────────

ALL_PITFALLS = {
    "1": pitfall_chunk_size,
    "2": pitfall_no_overlap,
    "3": pitfall_no_smoke_test,
    "4": pitfall_weak_prompt,
    "5": pitfall_vague_query,
}

DESCRIPTIONS = {
    "1": "Chunks too large",
    "2": "No chunk overlap",
    "3": "No retrieval smoke test",
    "4": "Weak prompt, no citations",
    "5": "Vague query design",
}


def main() -> None:
    args = sys.argv[1:]

    if not args or args[0] == "--list":
        print("\nAvailable pitfalls:")
        for k, v in DESCRIPTIONS.items():
            print(f"  {k}. {v}")
        print("\nUsage:")
        print("  python pitfalls_demo.py --all")
        print("  python pitfalls_demo.py 1")
        print("  python pitfalls_demo.py 1 2")
        return

    keys = list(ALL_PITFALLS.keys()) if args[0] == "--all" else [
        a for a in args if a in ALL_PITFALLS
    ]
    unknown = [a for a in args if a not in ALL_PITFALLS and a != "--all"]
    if unknown:
        print(f"\nUnknown pitfall(s): {unknown}")
        sys.exit(1)

    print(f"\n=== Pitfalls demo: {', '.join(DESCRIPTIONS[k] for k in keys)} ===")
    for k in keys:
        ALL_PITFALLS[k]()

    print("\n" + "-" * 68)
    print("Pitfall demos complete.")
    print("Compare the Weave traces for bad vs. good calls in your W&B project.")


if __name__ == "__main__":
    main()
