"""
Core RAG pipeline functions shared across all steps.

All model-facing operations are decorated with @weave.op() so they appear
in the Weave trace once weave.init() has been called by the calling script.
weave.init() does not need to be called here; the decorators activate tracing
automatically once the calling script initialises Weave.
"""

import os
import pickle

import faiss
import numpy as np
import weave
from openai import OpenAI

# ─── Constants ────────────────────────────────────────────────────────────────

INDEX_DIR = "index"
EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4o-mini"


# ─── Embedding ────────────────────────────────────────────────────────────────


@weave.op()
def embed_texts(texts: list[str], model: str = EMBEDDING_MODEL) -> list[list[float]]:
    """
    Embed a list of texts in one API call.
    Returns a list of raw (un-normalised) embedding vectors.
    Used during indexing (step 3); not used at query time.
    """
    client = OpenAI()
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


@weave.op()
def embed_query(query: str, model: str = EMBEDDING_MODEL) -> list[float]:
    """
    Embed a single query string and L2-normalise the result.
    Returns a plain Python list so Weave can serialise it for the trace log.
    """
    client = OpenAI()
    response = client.embeddings.create(input=[query], model=model)
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    vec = vec / max(float(np.linalg.norm(vec)), 1e-10)
    return vec.tolist()


# ─── Retrieval ────────────────────────────────────────────────────────────────


@weave.op()
def retrieve_chunks(query: str, top_k: int = 4) -> list[dict]:
    """
    Embed the query, search the FAISS index, and return the top_k results
    with similarity scores and source metadata.

    Requires step3_embed_index.py to have been run first so that
    index/faiss_index.bin and index/chunks.pkl exist.
    """
    index_path = os.path.join(INDEX_DIR, "faiss_index.bin")
    chunks_path = os.path.join(INDEX_DIR, "chunks.pkl")

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            "Run step3_embed_index.py first."
        )

    index = faiss.read_index(index_path)
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    # embed_query returns a list; convert to (1, dim) float32 array for FAISS
    query_vec = np.array(embed_query(query), dtype=np.float32).reshape(1, -1)
    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        chunk = chunks[idx]
        results.append(
            {
                "score": float(score),
                "source": chunk["source"],
                "chunk_id": chunk["chunk_id"],
                "text": chunk["text"],
            }
        )
    return results


# ─── Prompt assembly ──────────────────────────────────────────────────────────


def build_prompt(query: str, context_chunks: list[dict]) -> str:
    """
    Assemble the generation prompt from retrieved chunks.
    Not decorated with @weave.op() because it is pure string manipulation;
    its inputs and outputs are captured through generate_answer.
    """
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        context_parts.append(
            f"[{i}] Source: {chunk['source']}\n{chunk['text'].strip()}"
        )
    context_block = "\n\n---\n\n".join(context_parts)

    return (
        "You are a support assistant. Answer the user question using ONLY the context below.\n"
        "After each sentence that draws on a source, cite the filename in square brackets, "
        "for example: [image_policy.md].\n"
        'If the context does not contain enough information, say: '
        '"I cannot answer this from the provided context."\n\n'
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


# ─── Generation ───────────────────────────────────────────────────────────────


@weave.op()
def generate_answer(query: str, top_k: int = 4) -> dict:
    """
    Full RAG pipeline: retrieve context, build prompt, generate answer.

    Returns a dict with:
      - answer: the LLM's response string
      - sources_retrieved: list of unique source filenames used as context
      - context_chunks: the full list of retrieved chunk dicts (scores, text, etc.)
      - token_usage: prompt and completion token counts
    """
    context_chunks = retrieve_chunks(query, top_k=top_k)
    prompt = build_prompt(query, context_chunks)

    client = OpenAI()
    response = client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=512,
    )

    answer = response.choices[0].message.content
    sources_retrieved = list({c["source"] for c in context_chunks})

    return {
        "answer": answer,
        "sources_retrieved": sources_retrieved,
        "context_chunks": [
            {"score": c["score"], "source": c["source"], "chunk_id": c["chunk_id"]}
            for c in context_chunks
        ],
        "token_usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        },
    }
