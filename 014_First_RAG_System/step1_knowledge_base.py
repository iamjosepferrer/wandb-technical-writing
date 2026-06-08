"""
Step 1: Create a small knowledge base.

Writes five realistic support documents to a data/ directory.
These stand in for internal docs, help-centre articles, or engineering runbooks.
The sample corpus is embedded directly so you don't need any external files.

Usage:
    python step1_knowledge_base.py
"""

import os

OUTPUT_DIR = "data"

# ─── Document content ─────────────────────────────────────────────────────────
# Five documents with overlapping topics so retrieval has meaningful distinctions
# to make: images, retention, custom models, available models, and support FAQ.

DOCUMENTS: dict[str, str] = {
    "image_policy.md": """\
# Image Upload Policy

## Supported Formats
The platform accepts JPEG, PNG, GIF, WebP, and TIFF images. The maximum file size per
image is 20 MB. You can upload up to 50 images per API request.

## Allowed Use Cases
Images can be submitted alongside text prompts for vision-capable models. Common use
cases include document analysis, screenshot interpretation, diagram understanding, and
visual question answering.

## Restrictions
Images containing personally identifiable information (PII) are processed but not stored
beyond the session. Explicit content is filtered automatically. Images submitted via the
API are not used to train models unless you opt in to the data-sharing programme.

## Retention
Images submitted via the API are held in memory for the duration of the request and
deleted immediately after the response is returned. No image data is written to disk or
accessible to other users or sessions.

## Token Costs
Vision API calls count against the same token budget as text requests. A 512x512 image
uses approximately 85 tokens; a 1024x1024 image uses approximately 765 tokens.
Resolution-based tokenisation means large images can consume significant context.
""",
    "data_retention.md": """\
# Data Retention Policy

## Request Logs
All API request logs — including prompts, completions, metadata, and timestamps — are
retained for 30 days by default. Logs support abuse detection, rate limiting, and billing
reconciliation. They are not shared with third parties outside the processing chain.

## Extended Retention
Enterprise customers may request extended log retention up to 90 days by contacting
support. Extended retention applies to request metadata only; prompt and completion
content follows the default 30-day schedule unless a data processing agreement specifies
otherwise.

## Deletion Requests
You can request deletion of all data associated with your account via the account
dashboard. Production data is deleted within 72 hours; backup systems are purged within
30 days of the request.

## Compliance
Data retention practices comply with GDPR Article 5(1)(e) and CCPA Section 1798.105.
Enterprise customers receive a data processing addendum (DPA) covering subprocessor
lists, data transfer mechanisms, and audit rights.

## Embeddings and Vectors
Embeddings generated from your documents via the embedding API are stored only in your
own vector database. The platform does not retain copies of your embeddings or the source
documents used to create them.
""",
    "custom_llm_guide.md": """\
# Custom LLM Deployment Guide

## Overview
Custom LLMs are fine-tuned models trained on your data. Once a fine-tuning job finishes,
the resulting model is deployed to a dedicated endpoint that behaves identically to
standard model endpoints but reflects your domain-specific tuning.

## Supported Base Models
You can fine-tune gpt-4o-mini and gpt-3.5-turbo via the fine-tuning API. Custom LLM
training on gpt-4o is available for enterprise accounts on request.

## Training Data Requirements
Training files must be in JSONL format with a `messages` field, where each entry
contains a system turn, a user turn, and an assistant turn. A minimum of 50 training
examples is required to start a job; 500 to 1000 examples typically produce measurable
improvement over the base model.

## Deployment
After training completes the model ID appears in your fine-tuned models list. Use it in
API calls the same way as any base model ID. Custom LLM endpoints are hosted in the
same region as your account's primary data residency zone.

## Latency and Cost
Fine-tuned model inference costs 10 to 20 percent more per token than the equivalent
base model. Latency matches the base model for prompts under 2048 tokens and increases
proportionally for longer contexts.

## Updating a Custom LLM
Submit a new fine-tuning job with updated training data. The previous model endpoint
stays active until you delete it. You can run two endpoints in parallel for A/B testing
before switching production traffic.
""",
    "api_models.md": """\
# Available API Models

## Text Generation
- gpt-4o: Most capable model for text and vision tasks. Context window: 128k tokens.
  Supports structured outputs, function calling, and vision inputs.
- gpt-4o-mini: Smaller, faster, and cheaper than gpt-4o. Context window: 128k tokens.
  Recommended for high-volume tasks where full gpt-4o capability is not required.
- gpt-3.5-turbo: Legacy fast model. Context window: 16k tokens. Supported for existing
  integrations; gpt-4o-mini is preferred for new projects.

## Embeddings
- text-embedding-3-small: 1536-dimensional embeddings. Low cost, good accuracy for
  most semantic search and retrieval tasks. Recommended for new projects.
- text-embedding-3-large: 3072-dimensional embeddings. Higher accuracy for complex
  similarity tasks where precision matters more than cost.
- text-embedding-ada-002: Legacy embedding model. Use text-embedding-3-small for
  any new project.

## Image Generation
- dall-e-3: Generates 1024x1024, 1024x1792, or 1792x1024 images from text prompts.
  Supports vivid and natural quality modes.
- dall-e-2: Legacy image generation. Lower resolution than DALL-E 3.

## Rate Limits and Context Windows
Rate limits are set per model tier and are visible in your account dashboard. Enterprise
accounts can request higher limits via the quota-increase form. Context window limits
apply to the combined token count of the prompt and the completion.
""",
    "support_faq.md": """\
# Support FAQ

## Billing

Q: How is API usage billed?
Usage is billed per token. Input and output tokens are charged at different rates that
vary by model. Billing resets monthly; charges appear on your invoice two business days
after each billing period ends.

Q: Can I set a spending limit?
Yes. Set hard and soft limits in the billing settings page. A soft limit triggers an email
notification. A hard limit pauses API access for the remainder of the billing cycle or until
you raise the limit manually.

## Authentication

Q: How do I rotate my API key?
Generate a new key in the API keys section of the dashboard. The old key stays active
for 24 hours after you create the new one so you can rotate without downtime. After 24
hours the old key is automatically revoked.

Q: Can I have multiple API keys?
Yes. You can create up to 20 API keys per account. Each key can be labelled and scoped
to specific models or project IDs if your account uses project-based access control.

## Errors

Q: What does a 429 error mean?
A 429 status means you have exceeded your rate limit for the model or tier. Implement
exponential backoff starting at 1 second, doubling up to a maximum of 60 seconds. Most
rate limits reset within one minute.

Q: What is error code 503?
A 503 response means the model endpoint is temporarily unavailable, usually due to high
load. Retry with exponential backoff. If 503 errors persist for more than five minutes,
check the status page.
""",
}


def create_knowledge_base(output_dir: str = OUTPUT_DIR) -> list[dict]:
    os.makedirs(output_dir, exist_ok=True)
    records = []
    for filename, content in DOCUMENTS.items():
        filepath = os.path.join(output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        records.append(
            {
                "filename": filename,
                "path": filepath,
                "chars": len(content),
            }
        )
    return records


def main() -> None:
    print("=" * 50)
    print("Step 1: Create knowledge base")
    print("=" * 50)

    records = create_knowledge_base()

    print(f"\nCreated {len(records)} documents in {OUTPUT_DIR}/\n")
    print(f"{'Filename':<30}  {'Characters':>10}")
    print("-" * 44)
    for r in records:
        print(f"{r['filename']:<30}  {r['chars']:>10,}")

    total = sum(r["chars"] for r in records)
    print("-" * 44)
    print(f"{'Total':<30}  {total:>10,}")
    print("\nDone. Run step2_chunking.py next.")


if __name__ == "__main__":
    main()
