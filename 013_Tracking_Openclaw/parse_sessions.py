#!/usr/bin/env python3
"""
parse_sessions.py

Reads OpenClaw trajectory JSONL files, extracts token usage per conversation,
estimates costs, and logs each session as a Weave trace.

Usage:
    python parse_sessions.py
    python parse_sessions.py --days 7
    python parse_sessions.py --dry-run
    python parse_sessions.py --project my-project
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

MODEL_PRICING = {
    "gpt-4.1":       {"input": 2.00,  "output":  8.00},
    "gpt-4.1-mini":  {"input": 0.40,  "output":  1.60},
    "gpt-4o":        {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":   {"input": 0.15,  "output":  0.60},
    "gpt-5.5":       {"input": 6.00,  "output": 24.00},
    "claude-opus-4-6":   {"input": 15.00, "output": 75.00},
    "claude-sonnet-4-6": {"input":  3.00, "output": 15.00},
    "claude-haiku-4-5":  {"input":  0.80, "output":  4.00},
    "gemini-2.5-pro":    {"input":  1.25, "output": 10.00},
    "gemini-2.5-flash":  {"input":  0.15, "output":  0.60},
}

def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    # match on model name fragment
    for key, pricing in MODEL_PRICING.items():
        if key in (model_id or ""):
            return round(
                (input_tokens  / 1_000_000) * pricing["input"] +
                (output_tokens / 1_000_000) * pricing["output"],
                6
            )
    return 0.0

def parse_trajectory(filepath: Path) -> dict | None:
    total_input  = 0
    total_output = 0
    model_id     = None
    first_ts     = None
    last_ts      = None
    turns        = 0

    try:
        with open(filepath, encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    line = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                # Pick up model from any line that has it
                if not model_id and line.get("modelId"):
                    model_id = line["modelId"]

                # Token data lives in model.completed events
                if line.get("type") == "model.completed":
                    usage = line.get("data", {}).get("usage", {})
                    inp = usage.get("input", 0)
                    out = usage.get("output", 0)
                    if inp or out:
                        total_input  += inp
                        total_output += out
                        turns        += 1
                        ts = line.get("ts")
                        if ts:
                            if first_ts is None: first_ts = ts
                            last_ts = ts

    except (OSError, PermissionError) as e:
        print(f"  Warning: could not read {filepath.name}: {e}", file=sys.stderr)
        return None

    if turns == 0:
        return None

    return {
        "session_id":          filepath.stem.replace(".trajectory", ""),
        "model":               model_id or "unknown",
        "turns":               turns,
        "input_tokens":        total_input,
        "output_tokens":       total_output,
        "total_tokens":        total_input + total_output,
        "estimated_cost_usd":  estimate_cost(model_id or "", total_input, total_output),
        "first_message_at":    first_ts or "",
        "last_message_at":     last_ts  or "",
        "file_modified_at":    datetime.fromtimestamp(
                                   filepath.stat().st_mtime, tz=timezone.utc
                               ).isoformat(),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agents-dir", default=str(Path.home() / ".openclaw" / "agents"))
    parser.add_argument("--days",    type=int, default=None)
    parser.add_argument("--project", default="openclaw-costs")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    agents_dir = Path(args.agents_dir)
    if not agents_dir.exists():
        print(f"Error: {agents_dir} not found.", file=sys.stderr); sys.exit(1)

    # Read WANDB_API_KEY from ~/.openclaw/env if needed
    if not args.dry_run and not os.environ.get("WANDB_API_KEY"):
        env_file = Path.home() / ".openclaw" / "env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("WANDB_API_KEY="):
                    os.environ["WANDB_API_KEY"] = line.split("=", 1)[1].strip()
                    break
        if not os.environ.get("WANDB_API_KEY"):
            print("Error: WANDB_API_KEY not set.", file=sys.stderr); sys.exit(1)

    # Find trajectory files
    files = list(agents_dir.glob("**/*.trajectory.jsonl"))
    if args.days:
        import time
        cutoff = time.time() - args.days * 86400
        files = [f for f in files if f.stat().st_mtime >= cutoff]
    files = sorted(files, key=lambda f: f.stat().st_mtime)

    print(f"\nScanning: {agents_dir}")
    print(f"Found {len(files)} trajectory file(s). Parsing...\n")

    sessions = [s for s in (parse_trajectory(f) for f in files) if s]

    if not sessions:
        print("No usage data found. Have a few conversations first."); return

    print(f"{'Session':<38} {'Model':<18} {'Tokens':>8} {'Cost':>10}")
    print("-" * 78)
    for s in sessions:
        print(f"{s['session_id'][:36]:<38} {s['model'][:16]:<18} "
              f"{s['total_tokens']:>8,} ${s['estimated_cost_usd']:>9.6f}")

    total_tokens = sum(s["total_tokens"]       for s in sessions)
    total_cost   = sum(s["estimated_cost_usd"] for s in sessions)
    print("-" * 78)
    print(f"{'TOTAL':<38} {'':<18} {total_tokens:>8,} ${total_cost:>9.6f}\n")

    if args.dry_run:
        print("Dry run — no data sent to Weave."); return

    import weave
    weave.init(args.project)

    @weave.op()
    def log_session(session_id, model, turns, input_tokens, output_tokens,
                    total_tokens, estimated_cost_usd, first_message_at,
                    last_message_at, file_modified_at):
        return {
            "session_id": session_id, "model": model, "turns": turns,
            "input_tokens": input_tokens, "output_tokens": output_tokens,
            "total_tokens": total_tokens, "estimated_cost_usd": estimated_cost_usd,
            "first_message_at": first_message_at, "last_message_at": last_message_at,
            "file_modified_at": file_modified_at,
            "logged_at": datetime.now(timezone.utc).isoformat(),
        }

    print(f"Shipping {len(sessions)} session(s) to Weave project: {args.project}\n")
    for s in sessions:
        log_session(**s)
        print(f"  ✓ {s['session_id'][:50]}")

    print(f"\nDone. View at: https://wandb.ai/ai-team-articles/{args.project}/weave\n")

if __name__ == "__main__":
    main()
