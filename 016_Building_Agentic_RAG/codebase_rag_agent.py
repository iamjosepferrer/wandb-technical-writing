"""
Agentic RAG example 2: LLM-driven codebase Q&A.

Unlike the toy loop in agentic_rag_demo.py, everything here is model-driven:
- An LLM chooses which search tool to call and what arguments to pass.
- A separate LLM call reflects on accumulated evidence and decides whether
  to keep searching or stop.
- Nothing about the stopping criterion is hardcoded; it scales to any question.

Requirements:
    pip install -r requirements.txt  (openai added for this file)
    cp .env.example .env             # needs OPENAI_API_KEY and WANDB_API_KEY

Usage:
    python codebase_rag_agent.py
"""

import json
import os
import re
import tempfile

import weave
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# ── Embedded codebase ──────────────────────────────────────────────────────────
# A small fake Python project written to a temp directory so the demo runs
# without any real repository. Replace CODEBASE_ROOT in main() with a real
# project path to run against actual code.

FAKE_FILES = {
    "auth/middleware.py": """\
import hmac
import hashlib
from functools import wraps
from flask import request, abort
from auth.tokens import decode_token, TOKEN_SECRET


def require_auth(f):
    \"\"\"Decorator that validates the Bearer token on every protected route.\"\"\"
    @wraps(f)
    def decorated(*args, **kwargs):
        header = request.headers.get("Authorization", "")
        if not header.startswith("Bearer "):
            abort(401, "Missing Bearer token")
        token = header[7:]
        payload = decode_token(token)
        if payload is None:
            abort(401, "Invalid or expired token")
        request.user = payload
        return f(*args, **kwargs)
    return decorated


def sign_payload(payload: dict) -> str:
    \"\"\"HMAC-SHA256 signature over a JSON payload.\"\"\"
    raw = json.dumps(payload, sort_keys=True).encode()
    return hmac.new(TOKEN_SECRET.encode(), raw, hashlib.sha256).hexdigest()
""",
    "auth/tokens.py": """\
import jwt
import time

TOKEN_SECRET = "change-me-in-production"
TOKEN_TTL_SECONDS = 3600


def issue_token(user_id: str, roles: list) -> str:
    \"\"\"Issue a signed JWT valid for TOKEN_TTL_SECONDS.\"\"\"
    payload = {
        "sub": user_id,
        "roles": roles,
        "iat": int(time.time()),
        "exp": int(time.time()) + TOKEN_TTL_SECONDS,
    }
    return jwt.encode(payload, TOKEN_SECRET, algorithm="HS256")


def decode_token(token: str):
    \"\"\"Decode and validate a JWT. Returns None if invalid or expired.\"\"\"
    try:
        return jwt.decode(token, TOKEN_SECRET, algorithms=["HS256"])
    except Exception:
        return None
""",
    "api/routes.py": """\
from flask import Flask, jsonify, request
from auth.middleware import require_auth

app = Flask(__name__)


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/data")
@require_auth
def get_data():
    \"\"\"Protected endpoint; requires a valid Bearer token.\"\"\"
    return jsonify({"user": request.user["sub"], "data": [1, 2, 3]})


@app.route("/api/token", methods=["POST"])
def get_token():
    \"\"\"Issue a token for a user_id + roles payload.\"\"\"
    from auth.tokens import issue_token
    body = request.get_json()
    token = issue_token(body["user_id"], body.get("roles", []))
    return jsonify({"token": token})
""",
    "tests/test_auth.py": """\
import pytest
from api.routes import app
from auth.tokens import issue_token


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200


def test_protected_without_token(client):
    r = client.get("/api/data")
    assert r.status_code == 401


def test_protected_with_valid_token(client):
    token = issue_token("alice", ["admin"])
    r = client.get("/api/data", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert r.get_json()["user"] == "alice"
""",
}


def setup_codebase(root: str) -> None:
    """Write the fake codebase to a temporary directory."""
    for relative_path, content in FAKE_FILES.items():
        full_path = os.path.join(root, relative_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            f.write(content)


# ── Search tools ───────────────────────────────────────────────────────────────
# Three tools that mirror what grep, find, and cat do on a real filesystem.
# In production, swap these for subprocess calls to real bash commands,
# or point them at a code search API.

def grep_code(pattern: str, codebase_root: str, path: str = ".") -> str:
    """Search Python files for a regex pattern. Returns matching lines with filename:linenum."""
    matches = []
    search_root = os.path.join(codebase_root, path)
    for dirpath, _, filenames in os.walk(search_root):
        for fname in sorted(filenames):
            if not fname.endswith(".py"):
                continue
            fpath = os.path.join(dirpath, fname)
            rel = os.path.relpath(fpath, codebase_root)
            try:
                with open(fpath) as f:
                    for i, line in enumerate(f, 1):
                        if re.search(pattern, line, re.IGNORECASE):
                            matches.append(f"{rel}:{i}: {line.rstrip()}")
            except Exception:
                pass
    return "\n".join(matches) if matches else "No matches found."


def find_file(name_pattern: str, codebase_root: str) -> str:
    """Find files whose name matches a pattern."""
    results = []
    for dirpath, _, filenames in os.walk(codebase_root):
        for fname in sorted(filenames):
            if re.search(name_pattern, fname, re.IGNORECASE):
                rel = os.path.relpath(os.path.join(dirpath, fname), codebase_root)
                results.append(rel)
    return "\n".join(results) if results else "No files found."


def read_file(relative_path: str, codebase_root: str) -> str:
    """Read the full content of a file by its path relative to the codebase root."""
    full_path = os.path.join(codebase_root, relative_path)
    try:
        with open(full_path) as f:
            return f.read()
    except FileNotFoundError:
        return f"File not found: {relative_path}"


# ── Tool schema for OpenAI function calling ────────────────────────────────────
# The LLM selects from these tools and fills in the arguments itself.
# The parameter names must match the function signatures above.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "grep_code",
            "description": (
                "Search all Python files in the codebase for a regex pattern. "
                "Returns matching lines with filename and line number. "
                "Use this to locate function definitions, class names, imports, or keywords."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for (case-insensitive)",
                    },
                    "path": {
                        "type": "string",
                        "description": "Subdirectory to search within (default '.' = whole codebase)",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_file",
            "description": "Find files in the codebase whose name matches a regex pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name_pattern": {
                        "type": "string",
                        "description": "Regex pattern to match against filenames",
                    },
                },
                "required": ["name_pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full content of a file by its path relative to the codebase root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "relative_path": {
                        "type": "string",
                        "description": "File path relative to the codebase root, e.g. 'auth/middleware.py'",
                    },
                },
                "required": ["relative_path"],
            },
        },
    },
]


# ── LLM-driven reflection ──────────────────────────────────────────────────────
# This is the key difference from the toy example. Instead of a hardcoded rule
# that checks for specific keywords, an LLM reads all accumulated evidence and
# returns a structured judgment. It works for any question without any domain-
# specific code.

REFLECTION_SYSTEM = """\
You are a code search assistant deciding whether enough code has been retrieved to answer a question.

Rules:
- If you have read the FULL CONTENT of the key files (via read_file), you can declare sufficient.
- If you only have grep or find results (short matching lines, not full files), that is NOT sufficient.
  Set sufficient to false and name the specific file to read next in next_search.
- next_search must be concrete: "Read auth/middleware.py" not "find authentication code."
  If grep results mention a filename, that filename belongs in next_search.

Return a JSON object with exactly these fields:
{
  "sufficient": true or false,
  "reasoning": "one sentence explaining your judgment",
  "next_search": "concrete next action, e.g. 'Read auth/middleware.py for the full auth flow'"
}

Return ONLY the JSON object. No preamble, no markdown fences."""


@weave.op()
def reflect(question: str, evidence: list, step: int) -> dict:
    """
    LLM reflection step: decide whether the accumulated evidence is sufficient.

    Returns {"sufficient": bool, "reasoning": str, "next_search": str}.
    Called after every tool execution. When sufficient is True, the loop stops
    and generation begins.
    """
    evidence_text = "\n\n---\n\n".join(
        f"[{e['tool']}({e['args']})]\n{e['result']}"
        for e in evidence
    )
    user_msg = (
        f"Question: {question}\n\n"
        f"Search steps completed: {step}\n\n"
        f"Code retrieved so far:\n{evidence_text or '(nothing yet)'}"
    )
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": REFLECTION_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=200,
    )
    raw = resp.choices[0].message.content.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Defensive fallback: keep searching rather than generating from bad evidence
        return {"sufficient": False, "reasoning": "Could not parse reflection", "next_search": ""}


# ── LLM-driven planning ────────────────────────────────────────────────────────

PLANNING_SYSTEM = """\
You are a code search agent. Choose ONE tool call that will get you closer to answering the question.

Search strategy — follow this order:
1. Use grep_code or find_file to locate relevant files.
2. Once any search result mentions a specific filename, call read_file on that file immediately.
3. Do not run more grep searches when you already know which file to read.
4. Never repeat a search you have already done.

The most common mistake is continuing to grep when read_file on an identified file would give better results.
The reflection hint names the specific file or action to take next — follow it."""


@weave.op()
def plan_tool_call(question: str, evidence: list, next_search_hint: str) -> tuple:
    """
    LLM planning step: choose the next tool and its arguments.

    Uses OpenAI function calling with tool_choice="required" so the model
    always returns a tool call rather than a text response. The next_search_hint
    from the previous reflection step guides the choice without constraining it.
    """
    prior = "\n".join(f"- {e['tool']}({e['args']})" for e in evidence) or "(none yet)"
    user_msg = (
        f"Question: {question}\n\n"
        f"Searches done so far:\n{prior}\n\n"
        f"Reflection hint: {next_search_hint or 'Start fresh — no prior searches.'}"
    )
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": PLANNING_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        tools=TOOLS,
        tool_choice="required",  # force a tool call, never a plain text response
        temperature=0,
    )
    call = resp.choices[0].message.tool_calls[0]
    return call.function.name, json.loads(call.function.arguments)


# ── Tool dispatch ──────────────────────────────────────────────────────────────

@weave.op()
def run_tool(tool_name: str, tool_args: dict, codebase_root: str) -> str:
    """Execute a tool call and return the result as a string."""
    if tool_name == "grep_code":
        return grep_code(
            pattern=tool_args["pattern"],
            path=tool_args.get("path", "."),
            codebase_root=codebase_root,
        )
    if tool_name == "find_file":
        return find_file(tool_args["name_pattern"], codebase_root=codebase_root)
    if tool_name == "read_file":
        return read_file(tool_args["relative_path"], codebase_root=codebase_root)
    return f"Unknown tool: {tool_name}"


# ── Generation ─────────────────────────────────────────────────────────────────

GENERATION_SYSTEM = """\
You are a code assistant. Answer the question using ONLY the code snippets provided.
Cite the filename for each claim in square brackets, for example: [auth/middleware.py].
If the snippets are insufficient to answer fully, say so explicitly."""


@weave.op()
def generate_answer(question: str, evidence: list) -> str:
    """Generate the final answer from all accumulated code evidence."""
    evidence_text = "\n\n---\n\n".join(
        f"[{e['tool']}({e['args']})]\n{e['result']}"
        for e in evidence
    )
    client = OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": GENERATION_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Code snippets retrieved:\n\n{evidence_text}\n\n"
                    f"Question: {question}"
                ),
            },
        ],
        temperature=0,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


# ── Agent loop ─────────────────────────────────────────────────────────────────

@weave.op()
def codebase_rag_agent(question: str, codebase_root: str, max_steps: int = 6) -> dict:
    """
    Agentic RAG loop for codebase Q&A.

    Each iteration:
      1. plan_tool_call()  — LLM chooses which search tool to run and what arguments.
      2. run_tool()        — the tool executes and adds a result to evidence.
      3. reflect()         — LLM judges whether evidence is sufficient.
      4. If sufficient → generate_answer() and stop.
         Otherwise    → use next_search hint and repeat.

    max_steps is the hard budget: the loop stops regardless of the reflection
    judgment once this limit is reached.
    """
    state = {
        "question": question,
        "evidence": [],       # accumulated tool results
        "reflections": [],    # per-step LLM judgments
        "tool_calls": [],     # log of what was called at each step
        "answer": "",
    }
    next_search_hint = ""

    for step in range(max_steps):
        # Plan: LLM selects the next tool and its arguments
        tool_name, tool_args = plan_tool_call(question, state["evidence"], next_search_hint)
        state["tool_calls"].append({"step": step + 1, "tool": tool_name, "args": tool_args})

        # Execute: run the tool and record the result
        result = run_tool(tool_name, tool_args, codebase_root)
        state["evidence"].append({
            "tool": tool_name,
            "args": str(tool_args),
            "result": result,
        })

        # Reflect: LLM decides whether to keep searching or stop
        reflection = reflect(question, state["evidence"], step + 1)
        state["reflections"].append(reflection)

        if reflection.get("sufficient"):
            break

        next_search_hint = reflection.get("next_search", "")
        if not next_search_hint:
            # LLM has nothing more to suggest; stop rather than loop blindly
            break

    # Generate: build the final answer from all accumulated evidence
    state["answer"] = generate_answer(question, state["evidence"])
    return state


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    weave.init("agentic-rag-demo")

    with tempfile.TemporaryDirectory() as tmpdir:
        setup_codebase(tmpdir)

        question = (
            "How does authentication work in this codebase, "
            "and where is token validation implemented?"
        )
        print(f"Question: {question}\n")

        state = codebase_rag_agent(question, codebase_root=tmpdir)

        print("=== Answer ===")
        print(state["answer"])

        print(f"\nSearch steps: {len(state['tool_calls'])}")
        for tc in state["tool_calls"]:
            print(f"  Step {tc['step']}: {tc['tool']}({tc['args']})")

        print("\nReflections:")
        for i, r in enumerate(state["reflections"], 1):
            sufficient = r.get("sufficient", False)
            reasoning = r.get("reasoning", "")
            print(f"  Step {i}: sufficient={sufficient} — {reasoning}")