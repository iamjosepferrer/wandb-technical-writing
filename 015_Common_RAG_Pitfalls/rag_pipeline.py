"""
Core RAG pipeline using LangChain + Chroma + OpenAI.

Stack:
  - LangChain  : Document objects, RecursiveCharacterTextSplitter, retrieval
  - Chroma     : persistent local vector database
  - OpenAI     : text-embedding-3-small (embeddings), gpt-4o-mini (generation)
  - W&B Weave  : tracing and evaluation

All model-facing operations are decorated with @weave.op() so every
embedding call, retrieval query, and generation call appears as a traced
operation in the Weights & Biases Weave dashboard.

Functions that do not call an external API (chunking, loading) are not
decorated — they do not need to be traced.
"""

import os
import weave
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage

# ─── Constants ────────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4o-mini"
CHUNK_SIZE       = 1600   # characters — change in pitfalls_demo.py to see Pitfall 1
CHUNK_OVERLAP    = 320    # characters — set to 0 in pitfalls_demo.py to see Pitfall 2
RETRIEVAL_TOP_K  = 4
WANDB_PROJECT    = "rag-pitfalls"
CHROMA_DIR       = "chroma_db"
COLLECTION_NAME  = "rag_pitfalls"
DATA_DIR         = "docs"

# ─── Document loading ─────────────────────────────────────────────────────────

def load_documents(data_dir: str = DATA_DIR) -> list[Document]:
    """
    Load all .md and .txt files from data_dir as LangChain Document objects.

    Each Document carries a 'source' key in its metadata so retrieval results
    can be traced back to the originating file.
    """
    docs = []
    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith((".md", ".txt")):
            continue
        path = os.path.join(data_dir, fname)
        with open(path, encoding="utf-8") as f:
            text = f.read().strip()
        docs.append(Document(page_content=text, metadata={"source": fname}))
    return docs

# ─── Chunking ─────────────────────────────────────────────────────────────────

def chunk_documents(
    docs: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """
    Split documents into overlapping character-based chunks.

    Uses RecursiveCharacterTextSplitter, which tries to split on paragraph
    boundaries first, then sentence boundaries, then words. This avoids
    cutting mid-sentence wherever possible.

    Why overlap matters: without it, chunk boundaries are hard cuts. Any
    fact that spans two chunks gets split down the middle. A query targeting
    the second half of a concept will retrieve a chunk that starts mid-thought,
    missing the setup. Overlap carries a few sentences from the end of one
    chunk into the start of the next so boundary-spanning facts remain
    retrievable from either side.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        splits = splitter.split_text(doc.page_content)
        for i, split in enumerate(splits):
            chunks.append(Document(
                page_content=split,
                metadata={
                    "source":    source,
                    "chunk_id":  f"{source}__{i:02d}",
                    "char_count": len(split),
                },
            ))
    return chunks

# ─── Embedding and indexing ───────────────────────────────────────────────────

@weave.op()
def embed_and_store(
    chunks: list,
    collection_name: str = COLLECTION_NAME,
    persist_dir: str = CHROMA_DIR,
) -> dict:
    """
    Embed all chunks with OpenAI and store them in a Chroma collection.

    Returns a summary dict so Weave can log what was indexed.

    Note: the caller is responsible for deleting persist_dir before calling
    this function if a clean rebuild is needed (step3_embed_index.py does
    this automatically).
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_dir,
    )
    return {
        "chunks_indexed": len(chunks),
        "collection": collection_name,
        "persist_dir": persist_dir,
        "embedding_model": EMBEDDING_MODEL,
    }

# ─── Retrieval ────────────────────────────────────────────────────────────────

@weave.op()
def retrieve_docs(
    query: str,
    collection_name: str = COLLECTION_NAME,
    persist_dir: str = CHROMA_DIR,
    top_k: int = RETRIEVAL_TOP_K,
) -> list[dict]:
    """
    Embed the query and retrieve the top-k most similar chunks from Chroma.

    Chroma's similarity_search_with_relevance_scores returns scores in [0, 1]
    where 1.0 is most similar. Scores below 0.3 usually indicate that the
    retriever did not find a strong match — a useful diagnostic signal before
    connecting the generator (see Step 4).
    """
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    results = vectorstore.similarity_search_with_relevance_scores(query, k=top_k)
    return [
        {
            "page_content": doc.page_content,
            "source":       doc.metadata.get("source", "unknown"),
            "chunk_id":     doc.metadata.get("chunk_id", f"chunk_{i:02d}"),
            "score":        float(score),
        }
        for i, (doc, score) in enumerate(results)
    ]

# ─── Generation ───────────────────────────────────────────────────────────────

# Default (citation-enforcing) system prompt
SYSTEM_PROMPT = """\
You are a helpful assistant for a software platform.

Rules:
1. Answer ONLY from the provided context sections below.
2. After each factual claim, add a citation in the format [source: filename].
3. If the context does not contain the answer, respond with exactly:
   "I don't have information about that in the provided documentation."
4. Never invent information that is not in the context.

Context:
{context}"""

# Weak prompt used in step6_citations.py and pitfalls_demo.py to show Pitfall 4
WEAK_SYSTEM_PROMPT = "You are a helpful assistant. Answer the user's question."


@weave.op()
def generate_answer(
    query: str,
    retrieved_docs: list,
    system_prompt_template: str = SYSTEM_PROMPT,
) -> dict:
    """
    Generate a grounded answer using ChatOpenAI and the retrieved context.

    The system prompt template controls citation and refusal behaviour.
    Using SYSTEM_PROMPT enforces citations and out-of-scope refusals.
    Using WEAK_SYSTEM_PROMPT demonstrates Pitfall 4 (model fills gaps
    from training data instead of refusing).
    """
    if not retrieved_docs:
        return {
            "answer": "I don't have information about that in the provided documentation.",
            "sources": [],
        }

    context = "\n\n".join(
        f"[source: {d['source']}]\n{d['page_content']}" for d in retrieved_docs
    )

    llm = ChatOpenAI(model=GENERATION_MODEL, temperature=0.0)
    messages = [
        SystemMessage(content=system_prompt_template.format(context=context)),
        HumanMessage(content=query),
    ]
    response = llm.invoke(messages)

    answer  = response.content
    sources = list({d["source"] for d in retrieved_docs})
    return {"answer": answer, "sources": sources}


@weave.op()
def rag_query(
    query: str,
    collection_name: str = COLLECTION_NAME,
    persist_dir: str = CHROMA_DIR,
    top_k: int = RETRIEVAL_TOP_K,
) -> dict:
    """
    Full RAG pipeline: retrieve from Chroma, then generate with ChatOpenAI.

    Returns {query, answer, sources, retrieved_docs}.
    This function is used by RAGModel.predict() in step7_evaluate.py.
    """
    docs   = retrieve_docs(query, collection_name, persist_dir, top_k)
    result = generate_answer(query, docs)
    return {
        "query":         query,
        "answer":        result["answer"],
        "sources":       result["sources"],
        "retrieved_docs": docs,
    }
