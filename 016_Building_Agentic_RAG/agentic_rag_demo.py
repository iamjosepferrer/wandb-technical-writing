"""
Agentic RAG demo — minimal runnable loop with W&B Weave tracing.

The corpus is intentionally tiny so you can run this without any LLM API key.
Retrieval uses keyword overlap rather than embeddings, and reflection is
rule-based rather than model-driven. The goal is to make the loop structure
visible in your Weights & Biases project so you can inspect each step.

Requirements:
    pip install -r requirements.txt
    cp .env.example .env  # add your WANDB_API_KEY

Usage:
    python agentic_rag_demo.py
"""

import asyncio
import re
import weave
from dotenv import load_dotenv

load_dotenv()


# ── Corpus ─────────────────────────────────────────────────────────────────────
# Three documents representing different knowledge sources. A real system would
# index thousands of chunks from a vector store, SQL database, or document API.

DOCS = [
    {
        "id": "omdia-report",
        "source": "Omdia Report",
        "text": (
            "Omdia Report: enterprise AI teams adopting Agentic RAG need "
            "fresh external content, strong citations, and governance controls."
        ),
    },
    {
        "id": "support-messages",
        "source": "support messages",
        "text": (
            "Support messages: users complain that agentic workflows can add "
            "latency, retry too often, and cite irrelevant content when retrieval fails."
        ),
    },
    {
        "id": "rag-baseline",
        "source": "internal RAG notes",
        "text": (
            "Traditional RAG is fast and predictable for simple questions, but "
            "it struggles with multi-step queries and source selection."
        ),
    },
]


# ── Helper ─────────────────────────────────────────────────────────────────────

def _tokens(text: str) -> set[str]:
    """Lowercase token set used for overlap scoring."""
    return set(re.findall(r"[a-z0-9]+", text.lower()))


# ── Core loop functions — each decorated so Weave traces every call ─────────────

@weave.op()
def retrieve(query: str, exclude: list[str] | None = None, k: int = 1) -> list[dict]:
    """
    Toy retrieval: rank documents by keyword overlap with the query.

    In production this becomes a real search stack: dense vector search,
    BM25, a metadata filter, or a combination of all three. The @weave.op()
    decorator means every call is recorded with its inputs and outputs,
    so you can trace exactly what the agent retrieved at each step.
    """
    exclude_set = set(exclude or [])
    q = _tokens(query)
    scored = []

    for doc in DOCS:
        if doc["id"] in exclude_set:
            continue
        score = len(q & _tokens(f"{doc['source']} {doc['text']}"))
        if score:
            scored.append((score, doc))

    return [doc for _, doc in sorted(scored, reverse=True, key=lambda x: x[0])[:k]]


@weave.op()
def check_missing(question: str, evidence: list[dict]) -> list[str]:
    """
    Reflection step: identify which required sources are still absent.

    A real reflection function would send the question and accumulated evidence
    to a language model and ask it to identify gaps. Here we use keyword rules
    so the demo runs without an API key. The return value is a list of missing
    source types — an empty list means the agent has enough to answer.
    """
    joined = " ".join(f"{d['source']} {d['text']}" for d in evidence).lower()
    missing = []

    if "omdia" in question.lower() and "omdia" not in joined:
        missing.append("Omdia Report")
    if "support messages" in question.lower() and "support messages" not in joined:
        missing.append("support messages")
    if "latency" in question.lower() and "latency" not in joined:
        missing.append("latency evidence")

    return missing


@weave.op()
def plan_next_query(missing: list[str]) -> str:
    """
    Planning step: choose the next retrieval query based on what is still needed.

    A real planner would use an LLM to reformulate or decompose the question.
    Here it's a simple lookup so you can see the planning decision as a distinct
    traced step separate from retrieval.
    """
    if "support messages" in missing:
        return "support messages agentic RAG latency complaints"
    if "Omdia Report" in missing:
        return "Omdia Report Agentic RAG governance citations"
    if "latency evidence" in missing:
        return "agentic workflows latency retries"
    return ""


@weave.op()
def agentic_rag(question: str, max_steps: int = 3) -> dict:
    """
    Minimal agentic RAG loop.

    Observe → retrieve → reflect → plan → repeat, until the reflection step
    says evidence is sufficient or the step budget runs out.

    Returns the full state dict: question, per-step queries, accumulated
    evidence, reflections, and the final answer. In Weave you will see this
    as the root span, with retrieve, check_missing, and plan_next_query
    as child spans for each iteration.
    """
    state: dict = {
        "question": question,
        "queries": [],
        "evidence": [],
        "reflections": [],
        "answer": "",
    }
    query = question
    seen: set[str] = set()

    for step in range(max_steps):
        state["queries"].append(query)

        # Retrieve — exclude sources already in evidence to avoid duplication
        docs = retrieve(query, exclude=list(seen))
        for doc in docs:
            state["evidence"].append(doc)
            seen.add(doc["id"])

        # Reflect — check whether we have everything the question requires
        missing = check_missing(question, state["evidence"])
        state["reflections"].append(
            "sufficient" if not missing else f"missing: {', '.join(missing)}"
        )

        if not missing:
            # Evidence is complete; stop the loop early
            break

        # Plan — decide what to look for next
        next_query = plan_next_query(missing)
        if not next_query:
            # Planner has nothing to suggest; stop rather than loop indefinitely
            break
        query = next_query

    # Build the final answer from accumulated evidence
    citations = ", ".join(doc["id"] for doc in state["evidence"])
    state["answer"] = (
        "Prioritise governance, citation quality, and latency budgets. "
        "The analyst evidence emphasises fresh content and controls; "
        "support messages reveal operational risk from retries and irrelevant citations. "
        f"Sources: {citations}"
    )
    return state


# ── Weave Model and Evaluation ──────────────────────────────────────────────────
# The AgenticRAGModel wraps the loop so weave.Evaluation can call it as a
# standard predict() method. The scorers check source coverage (did the agent
# find all required documents?) and loop efficiency (did it stay within budget?).

class AgenticRAGModel(weave.Model):
    max_steps: int = 3

    @weave.op()
    def predict(self, question: str) -> dict:
        state = agentic_rag(question, self.max_steps)
        return {
            "answer": state["answer"],
            "sources": [doc["source"] for doc in state["evidence"]],
            "loop_count": len(state["queries"]),
        }


@weave.op()
def source_coverage_scorer(output: dict, expected_sources: list) -> dict:
    """
    Check whether all required sources appear in the retrieved evidence.

    The parameter names matter: 'output' receives the model's return value,
    and 'expected_sources' matches the column name in the evaluation dataset
    so Weave can pass the right value automatically.
    """
    retrieved = set(output.get("sources", []))
    expected = set(expected_sources)
    hits = len(retrieved & expected)
    coverage = hits / max(len(expected), 1)
    return {
        "coverage": round(coverage, 2),
        "all_found": coverage == 1.0,
    }


@weave.op()
def loop_efficiency_scorer(output: dict, max_loops: int) -> dict:
    """Flag loops that exceeded the expected step budget for the question."""
    count = output.get("loop_count", 0)
    return {
        "loop_count": count,
        "within_budget": count <= max_loops,
    }


# Four evaluation examples — enough to see patterns in the W&B Evaluations view
EVAL_DATASET = [
    {
        "question": (
            "Using the Omdia Report and support messages, "
            "which Agentic RAG risks should we prioritise for latency?"
        ),
        "expected_sources": ["Omdia Report", "support messages"],
        "max_loops": 3,
    },
    {
        "question": "What does the Omdia Report say about governance?",
        "expected_sources": ["Omdia Report"],
        "max_loops": 2,
    },
    {
        "question": "What are the support messages complaints about?",
        "expected_sources": ["support messages"],
        "max_loops": 2,
    },
    {
        "question": (
            "How does traditional RAG compare to agentic RAG for multi-step queries?"
        ),
        "expected_sources": ["internal RAG notes"],
        "max_loops": 2,
    },
]


async def run_evaluation() -> None:
    model = AgenticRAGModel(max_steps=3)
    evaluation = weave.Evaluation(
        name="agentic_rag_loop_eval",
        dataset=EVAL_DATASET,
        scorers=[source_coverage_scorer, loop_efficiency_scorer],
    )
    results = await evaluation.evaluate(model)
    print("\nEvaluation summary:", results)


# ── Entry point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    weave.init("agentic-rag-demo")

    # ── Single-query trace ───────────────────────────────────────────────────
    # Run this first; open the printed URL to see the full trace tree in
    # the W&B Weave UI before looking at aggregate evaluation results.
    print("Running single-query trace...\n")
    state = agentic_rag(
        "Using the Omdia Report and support messages, "
        "which Agentic RAG risks should we prioritise for latency?"
    )
    print(f"Answer  : {state['answer']}")
    print(f"Loops   : {len(state['queries'])}")
    print(f"Sources : {[d['source'] for d in state['evidence']]}")
    print(f"\nReflections per step:")
    for i, (q, r) in enumerate(zip(state["queries"], state["reflections"]), 1):
        print(f"  Step {i}: query='{q[:60]}...'  reflection='{r}'")

    # ── Evaluation ──────────────────────────────────────────────────────────
    # Runs all four examples and logs scores to W&B Evaluations.
    print("\nRunning evaluation against all examples...")
    asyncio.run(run_evaluation())
    print("\nDone. Open your Weights & Biases project to inspect traces and scores.")
