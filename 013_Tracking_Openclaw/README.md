# 013 — Tracking OpenClaw costs and setting guardrails with W&B Weave

Code for the Weights & Biases tutorial.

---

## How it works

OpenClaw stores every conversation in JSONL files at `~/.openclaw/agents/main/sessions/`. Each assistant turn records the model used and the token counts for that call.

`parse_sessions.py` reads those files, aggregates tokens and cost per session, and ships each session as a Weave trace to your Weights & Biases project. Run it from your terminal after any set of conversations.

---

## Files

```
013_Tracking_Openclaw/
├── parse_sessions.py           # Reads session logs, ships to W&B Weave
├── setup.sh                    # Installs dependencies
├── requirements.txt
└── config/
    └── openclaw.json.example   # Guardrails + W&B Inference provider
```

---

## Setup

```bash
bash setup.sh
```

This creates a `.venv` inside the project folder, installs `wandb` and `weave`, and saves your `WANDB_API_KEY` to `~/.openclaw/env`.

---

## Usage

**1. Have some conversations** in Telegram with your OpenClaw agent.

**2. Dry-run first** to see what will be logged:

```bash
.venv/bin/python parse_sessions.py --dry-run
```

**3. Ship to Weave:**

```bash
.venv/bin/python parse_sessions.py
```

**4. View your traces:**

```
https://wandb.ai/YOUR_ENTITY/openclaw-costs/weave
```

---

## Options

```
--days N          Only include sessions modified in the last N days
--project NAME    Weave project name (default: openclaw-costs)
--agents-dir PATH Custom path to OpenClaw agents directory
--dry-run         Parse and print without sending to Weave
```

---

## Comparing models

Switch models mid-session in Telegram:

```
/model openai/gpt-4o
```

Run the same task again. Then run `parse_sessions.py` — each session appears as a separate trace in Weave with its model, token counts, and cost. Sort by `estimated_cost_usd` to see the difference.

---

## Guardrails

Copy the relevant blocks from `config/openclaw.json.example` into `~/.openclaw/openclaw.json`:

| Setting | What it does |
|---|---|
| `channels.telegram.allowFrom` | Allowlist of Telegram user IDs |
| `agents.defaults.contextTokens` | Cap context window tokens per session |
| `agents.defaults.maxConcurrent` | Max parallel LLM requests |

Run `openclaw gateway restart` after any config change.

---

## W&B Inference as a provider

Add the `wandb-inference` block from `config/openclaw.json.example` to your `models.providers` section. Replace `YOUR_WANDB_MODEL_ID` with the model ID from your W&B Inference deployment.

Switch to it in Telegram:
```
/model wandb-inference/YOUR_MODEL_ID
```

Run `parse_sessions.py` after a few conversations to compare costs against your OpenAI baseline in the same Weave project.
