"""
Step 3: Embed chunks and build the FAISS vector index.

This is the first step that calls the OpenAI API, so Weave tracing starts here.
embed_texts() in rag_pipeline.py is decorated with @weave.op(); initialising Weave
before calling it means the embedding batch appears in the trace.

Saves:
  index/faiss_index.bin  — the FAISS flat inner-product index
  index/chunks.pkl       — the chunk list (same order as index rows)

Usage:
    python step3_embed_index.py
"""

import json
import os
import pickle

import faiss
import numpy as np
import weave
from dotenv import load_dotenv

from rag_pipeline import EMBEDDING_MODEL, embed_texts

load_dotenv()

CHUNKS_PATH = "chunks.json"
INDEX_DIR = "index"
WANDB_PROJECT = "rag-tutorial"
BATCH_SIZE = 100   # stay well within the OpenAI embeddings API limit


# ─── Helper functions ─────────────────────────────────────────────────────────


def normalize(vectors: np.ndarray) -> np.ndarray:
    """
    L2-normalise each row so that FAISS inner-product search
    is equivalent to cosine similarity.
    """
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, 1e-10, None)


def build_faiss_index(vectors: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a flat inner-product index.
    IndexFlatIP is exact (brute-force) — perfect for a tutorial corpus.
    For larger collections (> ~100k chunks), switch to IndexIVFFlat or
    IndexHNSWFlat for faster approximate search.
    """
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    # Initialise Weave before any model-facing call so the embedding trace
    # appears in the Weave project dashboard.
    weave.init(WANDB_PROJECT)

    print("=" * 50)
    print("Step 3: Embed chunks and build vector index")
    print("=" * 50)

    if not os.path.exists(CHUNKS_PATH):
        print(f"\nERROR: {CHUNKS_PATH} not found. Run step2_chunking.py first.")
        return

    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"\nLoaded {len(chunks)} chunks from {CHUNKS_PATH}")

    # ─── Embed in batches ─────────────────────────────────────────────────────
    texts = [c["text"] for c in chunks]
    raw_vectors: list[list[float]] = []

    print(f"Embedding with {EMBEDDING_MODEL} (batch size {BATCH_SIZE})...")
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        batch_vecs = embed_texts(batch)   # @weave.op — appears in Weave trace
        raw_vectors.extend(batch_vecs)
        print(f"  Batch {i // BATCH_SIZE + 1}: {len(batch)} chunks embedded")

    # ─── Normalise and index ──────────────────────────────────────────────────
    vectors = normalize(np.array(raw_vectors, dtype=np.float32))
    print(f"\nEmbedding shape : {vectors.shape}")      # e.g. (18, 1536)
    print(f"Embedding dim   : {vectors.shape[1]}")

    index = build_faiss_index(vectors)
    print(f"FAISS ntotal    : {index.ntotal}")

    # ─── Persist ──────────────────────────────────────────────────────────────
    os.makedirs(INDEX_DIR, exist_ok=True)
    index_path = os.path.join(INDEX_DIR, "faiss_index.bin")
    chunks_path = os.path.join(INDEX_DIR, "chunks.pkl")

    faiss.write_index(index, index_path)
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)

    print(f"\nSaved index  -> {index_path}")
    print(f"Saved chunks -> {chunks_path}")
    print("\nDone. Run step4_retrieve.py next.")


if __name__ == "__main__":
    main()
