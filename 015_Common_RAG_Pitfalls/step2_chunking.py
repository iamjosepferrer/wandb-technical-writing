"""
Step 2: Split documents into retrievable sections.

Uses LangChain's RecursiveCharacterTextSplitter, which tries to split on
paragraph boundaries first, then sentence boundaries, then words. This
is the recommended splitter for most documentation-style corpora.

No API keys required.

Usage:
    python step2_chunking.py
"""

import json
import os
import sys

from rag_pipeline import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    chunk_documents,
    load_documents,
)

CHUNKS_PATH = "chunks.json"


def show_chunk_comparison(docs: list) -> None:
    """Print a side-by-side comparison of three chunking strategies."""
    configs = [
        ("TOO LARGE  (chunk_size=8000, overlap=0)",     8000, 0),
        ("NO OVERLAP (chunk_size=1600, overlap=0)",     1600, 0),
        ("CORRECT    (chunk_size=1600, overlap=320)",   1600, 320),
    ]

    print("Chunking strategy comparison\n")
    print(f"  {'Strategy':<46}  {'Chunks':>6}  {'Avg chars':>9}")
    print(f"  {'-'*46}  {'-'*6}  {'-'*9}")

    for label, cs, ov in configs:
        chunks = chunk_documents(docs, chunk_size=cs, chunk_overlap=ov)
        avg = sum(c.metadata["char_count"] for c in chunks) / len(chunks)
        print(f"  {label:<46}  {len(chunks):>6}  {avg:>9.0f}")

    print()
    print("Why it matters:")
    print("  TOO LARGE : retriever returns entire documents, not specific passages.")
    print("              The LLM gets flooded with irrelevant context.")
    print("  NO OVERLAP: ideas at chunk boundaries are cut in half. A question about")
    print("              training requirements may retrieve a chunk that ends just")
    print("              before the key sentence.")
    print("  CORRECT   : focused passages with a few sentences of shared context")
    print("              between neighbouring chunks.\n")


def show_boundary_example(docs: list) -> None:
    """
    Show a concrete boundary example with and without overlap.

    Uses image_policy.md because it is long enough to produce multiple
    chunks at chunk_size=1600 and has a clear section boundary.
    """
    target = next((d for d in docs if d.metadata["source"] == "image_policy.md"), docs[0])

    no_ov   = chunk_documents([target], chunk_size=1600, chunk_overlap=0)
    with_ov = chunk_documents([target], chunk_size=1600, chunk_overlap=320)

    if len(no_ov) < 2:
        return

    print("Boundary example — image_policy.md\n")
    print("  WITHOUT overlap (overlap=0):")
    print(f"    end of chunk 0 :  ...{no_ov[0].page_content[-130:].replace(chr(10), ' ').strip()}")
    print(f"    start of chunk 1:   {no_ov[1].page_content[:130].replace(chr(10), ' ').strip()}...")

    if len(with_ov) >= 2:
        print()
        print("  WITH overlap (overlap=320):")
        print(f"    end of chunk 0 :  ...{with_ov[0].page_content[-130:].replace(chr(10), ' ').strip()}")
        print(f"    start of chunk 1:   {with_ov[1].page_content[:130].replace(chr(10), ' ').strip()}...")

    print()
    print("  The overlapping version repeats a few sentences between chunks so a")
    print("  query can find the right context even when it spans a boundary.\n")


def main() -> None:
    print("\n=== Step 2: Chunk documents ===\n")

    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: {DATA_DIR}/ not found. Run step1_knowledge_base.py first.")
        sys.exit(1)

    docs = load_documents(DATA_DIR)
    print(f"Loaded {len(docs)} documents from {DATA_DIR}/\n")

    show_chunk_comparison(docs)
    show_boundary_example(docs)

    # Save the correct chunks for step3
    chunks = chunk_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    # Serialise to JSON for inspection (Chroma receives Document objects in step3,
    # but having a readable JSON copy is useful for debugging chunk quality)
    chunks_data = [
        {
            "chunk_id":   c.metadata["chunk_id"],
            "source":     c.metadata["source"],
            "char_count": c.metadata["char_count"],
            "text":       c.page_content,
        }
        for c in chunks
    ]
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks_data, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(chunks)} chunks to {CHUNKS_PATH}")
    print(f"  chunk_size={CHUNK_SIZE} chars, overlap={CHUNK_OVERLAP} chars\n")

    col_id  = 32
    col_src = 26
    col_cnt = 7
    header = f"  {'Chunk ID':<{col_id}}  {'Source':<{col_src}}  {'Chars':>{col_cnt}}  Preview"
    print(header)
    print("  " + "-" * (col_id + col_src + col_cnt + 6 + 50))

    for c in chunks_data:
        preview = c["text"].replace("\n", " ")[:60].strip()
        print(
            f"  {c['chunk_id']:<{col_id}}  "
            f"{c['source']:<{col_src}}  "
            f"{c['char_count']:>{col_cnt}}  "
            f"{preview}..."
        )

    avg = sum(c["char_count"] for c in chunks_data) / len(chunks_data)
    print(f"\n  Total chunks : {len(chunks_data)}")
    print(f"  Average size : {avg:.0f} chars")
    print("\nDone. Run step3_embed_index.py next.")


if __name__ == "__main__":
    main()
