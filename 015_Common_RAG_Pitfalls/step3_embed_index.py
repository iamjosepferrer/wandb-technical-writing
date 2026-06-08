"""
Step 3: Embed document sections and store them in Chroma.

This is the first step that calls the OpenAI API, so Weave tracing
begins here. Every call to embed_and_store(), retrieve_docs(),
generate_answer(), and rag_query() is decorated with @weave.op() and
will appear as a named trace in your Weights & Biases project.

Chroma persists the vector store to chroma_db/. The directory is deleted
and rebuilt on each run so you always start from a clean index. If you
change EMBEDDING_MODEL in rag_pipeline.py, simply re-run this step.

Usage:
    python step3_embed_index.py

Requires:
    OPENAI_API_KEY in .env (or environment)
    docs/ created by step1_knowledge_base.py
    chunks.json created by step2_chunking.py  (for doc count reference)
"""

import os
import shutil
import sys

import weave
from dotenv import load_dotenv

from rag_pipeline import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
    WANDB_PROJECT,
    chunk_documents,
    embed_and_store,
    load_documents,
)

load_dotenv()


def main() -> None:
    print("\n=== Step 3: Embed sections and build Chroma index ===\n")

    if not os.path.isdir(DATA_DIR):
        print(f"ERROR: {DATA_DIR}/ not found. Run step1_knowledge_base.py first.")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not set. Check your .env file.")
        sys.exit(1)

    # Load and chunk documents
    docs   = load_documents(DATA_DIR)
    chunks = chunk_documents(docs)
    print(f"Loaded {len(docs)} documents, split into {len(chunks)} chunks")

    # Delete existing index so re-runs are always clean
    if os.path.exists(CHROMA_DIR):
        shutil.rmtree(CHROMA_DIR)
        print(f"Removed existing {CHROMA_DIR}/ for a clean rebuild")

    # Initialise Weave — all @weave.op() decorated functions log from here
    weave.init(WANDB_PROJECT)
    print(f"Weave initialised, project: {WANDB_PROJECT}\n")

    # Embed and store — this call itself is traced by Weave
    result = embed_and_store(chunks, collection_name=COLLECTION_NAME, persist_dir=CHROMA_DIR)

    print(f"Index saved to {CHROMA_DIR}/")
    print(f"  Chunks indexed    : {result['chunks_indexed']}")
    print(f"  Collection        : {result['collection']}")
    print(f"  Embedding model   : {EMBEDDING_MODEL}")
    print()
    print("Important: if you change EMBEDDING_MODEL in rag_pipeline.py, re-run")
    print(f"  this step to rebuild {CHROMA_DIR}/ with the new model.")
    print("\nDone. Run step4_smoke_test.py next.")


if __name__ == "__main__":
    main()
