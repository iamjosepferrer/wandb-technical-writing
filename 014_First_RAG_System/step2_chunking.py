"""
Step 2: Chunk documents and inspect what the retriever will see.

Reads documents from data/, splits them into overlapping token-based chunks,
and prints a table showing chunk IDs, sizes, and text previews.
This is the first quality checkpoint: bad chunks lead to bad retrieval
even with strong embeddings.

Saves chunks to chunks.json for use by step3_embed_index.py.

Usage:
    python step2_chunking.py
"""

import json
import os

import tiktoken

DATA_DIR = "data"
OUTPUT_PATH = "chunks.json"

# ─── Chunking parameters ──────────────────────────────────────────────────────
# 400 tokens with 80-token overlap works well for documentation-style text.
# Smaller chunks improve precision but can cut off context mid-thought.
# Larger chunks preserve context but dilute similarity scores and fill the
# context window faster at generation time.

CHUNK_SIZE = 400      # tokens per chunk
CHUNK_OVERLAP = 80    # tokens shared between consecutive chunks


# ─── Functions ────────────────────────────────────────────────────────────────


def load_documents(data_dir: str) -> list[dict]:
    docs = []
    for filename in sorted(os.listdir(data_dir)):
        if not filename.endswith(".md"):
            continue
        filepath = os.path.join(data_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        docs.append({"source": filename, "content": content})
    return docs


def chunk_document(
    doc: dict,
    enc: tiktoken.Encoding,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    """
    Split a single document into overlapping chunks.
    Each chunk carries a unique ID and source metadata so the retriever
    can trace every result back to its origin document.
    """
    tokens = enc.encode(doc["content"])
    chunks = []
    chunk_idx = 0
    start = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)

        chunks.append(
            {
                "chunk_id": f"{doc['source']}__chunk{chunk_idx:03d}",
                "source": doc["source"],
                "token_count": len(chunk_tokens),
                "text": chunk_text,
            }
        )

        if end == len(tokens):
            break
        start += chunk_size - overlap
        chunk_idx += 1

    return chunks


def chunk_all_documents(docs: list[dict]) -> list[dict]:
    # cl100k_base is the tokeniser used by text-embedding-3-small and gpt-4o
    enc = tiktoken.get_encoding("cl100k_base")
    all_chunks = []
    for doc in docs:
        all_chunks.extend(chunk_document(doc, enc, CHUNK_SIZE, CHUNK_OVERLAP))
    return all_chunks


def main() -> None:
    print("=" * 50)
    print("Step 2: Chunk documents")
    print("=" * 50)

    if not os.path.isdir(DATA_DIR):
        print(f"\nERROR: {DATA_DIR}/ not found. Run step1_knowledge_base.py first.")
        return

    docs = load_documents(DATA_DIR)
    print(f"\nLoaded {len(docs)} documents from {DATA_DIR}/")

    chunks = chunk_all_documents(docs)

    # Save for step 3
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(chunks)} chunks to {OUTPUT_PATH}\n")

    # ─── Inspection table ─────────────────────────────────────────────────────
    col_id = 38
    col_src = 26
    col_tok = 7
    header = f"{'Chunk ID':<{col_id}}  {'Source':<{col_src}}  {'Tokens':>{col_tok}}  Preview"
    print(header)
    print("-" * (len(header) + 20))
    for c in chunks:
        preview = c["text"].replace("\n", " ")[:80].strip()
        print(
            f"{c['chunk_id']:<{col_id}}  "
            f"{c['source']:<{col_src}}  "
            f"{c['token_count']:>{col_tok}}  "
            f"{preview}..."
        )

    print(f"\nTotal chunks : {len(chunks)}")
    avg = sum(c["token_count"] for c in chunks) / len(chunks)
    print(f"Average size : {avg:.0f} tokens")
    print(
        "\nIf chunk previews look broken mid-sentence, reduce CHUNK_SIZE "
        "or increase CHUNK_OVERLAP."
    )
    print("Done. Run step3_embed_index.py next.")


if __name__ == "__main__":
    main()
