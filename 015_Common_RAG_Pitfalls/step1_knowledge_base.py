"""
Step 1: Create the local knowledge base.

Writes five Markdown documents about a fictional SaaS platform to docs/.
These are the same documents used throughout every subsequent step.

No API keys required — this step is pure file I/O.

Usage:
    python step1_knowledge_base.py

Common pitfall prevented here:
    Documents without clear headings or structure produce poor chunks.
    Every file here uses H2 headings to create natural section boundaries
    that RecursiveCharacterTextSplitter can work with.
"""

import os

DATA_DIR = "docs"

DOCUMENTS = {
    "image_policy.md": """\
# Image Upload Policy

## Supported Formats

The platform accepts the following image formats: PNG, JPEG, GIF, WebP, and BMP.
PNG is recommended for screenshots and diagrams that require transparency or lossless
compression. JPEG works best for photographs and images with complex color gradients.
GIF is limited to 256 colors and is suitable for simple animations. WebP provides
superior compression compared to JPEG and PNG at equivalent quality and is the
recommended format for web assets. BMP is accepted but not recommended because its
uncompressed nature produces large file sizes.

## File Size Limits

The maximum file size for a single image upload is 10 MB. Files that exceed this limit
are automatically rejected with a 413 error code. Attempting to resize a file
client-side before upload is acceptable but not required for files within the size
limit. The platform does not impose a monthly storage quota for the Standard and
Professional plans. Enterprise plans have configurable storage quotas managed through
the admin console.

## Batch Uploads

Developers can upload up to 20 images in a single API call using the batch endpoint at
POST /v1/images/batch. Each image in the batch is processed independently, which means
a single failure does not block the remaining uploads. The batch endpoint returns a list
of results with status codes for each item. Successful uploads return a 201 status code
and an image ID. Failed uploads include an error code and message.

## Image Retention

Uploaded images are stored for 90 days from the date of upload. After 90 days, images
are permanently deleted and cannot be recovered. If your application requires permanent
storage, download images before the retention window expires and store them in your own
infrastructure. The platform sends an email notification 7 days before each image
expires if email notifications are enabled for the account.
""",

    "data_retention.md": """\
# Data Retention Policy

## API Request Logs

The platform retains API request logs for 30 days. Request logs include the endpoint
called, timestamp, request headers, and anonymized input metadata. Response content is
not stored in request logs. After 30 days, logs are permanently deleted and cannot be
recovered. Enterprise customers can request extended log retention of up to 12 months
through a support ticket. Extended retention is billed at a per-gigabyte monthly rate.

## Model Outputs

Model outputs are not stored by default. If your application needs to reference
previous model outputs, implement storage in your own database. For debugging and
audit purposes, developers can enable response logging through the API settings panel.
When response logging is enabled, outputs are retained for 7 days and then permanently
deleted. Enabling response logging counts toward your storage quota.

## User Account Data

User account data, including email addresses, billing information, and profile settings,
is retained for the lifetime of the account. When an account is deleted, all personally
identifiable information is removed within 30 days in compliance with GDPR Article 17.
API keys associated with a deleted account are immediately revoked. Usage statistics
are retained in anonymized form after account deletion.

## GDPR and Compliance

The platform is GDPR-compliant. Users in the European Union can submit data access
requests through the privacy portal. Data export requests are fulfilled within 72 hours.
Data deletion requests are processed within 30 days. If you process personal data
through the platform on behalf of EU data subjects, you should review the Data
Processing Agreement available in the legal section of your account settings.
""",

    "custom_llm_guide.md": """\
# Custom LLM Fine-Tuning Guide

## Minimum Training Examples

A custom model requires a minimum of 50 training examples to start the fine-tuning
process. This is the absolute minimum; models trained on 50 examples will show limited
improvement over the base model. For reliable performance gains, a dataset of 100 or
more high-quality examples is recommended. High quality means diverse, well-formatted
examples that cover the range of inputs the model will encounter in production.
Quantity alone does not guarantee quality. A dataset of 50 carefully curated examples
consistently outperforms a dataset of 200 noisy examples.

## Training Data Format

Training data must be submitted in JSONL format. Each line in the file is a valid JSON
object with a messages field that follows the standard chat completion format. The file
cannot exceed 100 MB in size. Files that fail format validation are rejected before
billing begins. A dry-run endpoint is available at POST /v1/fine-tunes/validate to
check your file before submitting a full training job.

## Base Models

Fine-tuning is supported on the following base models: gpt-4o-mini and gpt-3.5-turbo.
Larger models are not available for fine-tuning. The choice of base model should be
driven by your latency and cost requirements. gpt-4o-mini produces stronger results for
complex reasoning tasks, while gpt-3.5-turbo is faster and cheaper if your use case is
straightforward.

## Training Duration and Cost

A training job for 100 examples typically completes in approximately two hours.
Training time scales roughly linearly with dataset size. Fine-tuning is billed at
$0.003 per training token. The training token count equals the total tokens across all
examples multiplied by the number of epochs, which defaults to three. You can override
the epoch count with the n_epochs parameter, though reducing epochs below three is not
recommended for datasets under 500 examples.

## Evaluating a Fine-Tuned Model

The platform automatically holds out 10 percent of your training data for evaluation.
After training completes, the evaluation results appear in the fine-tuning dashboard,
including accuracy, loss curve, and sample completions. You should also run your own
offline evaluation on a separate held-out test set before deploying a fine-tuned model
to production.
""",

    "support_faq.md": """\
# Support FAQ

## Understanding API Error Codes

A 429 error means your application has exceeded the allowed number of requests per
minute or tokens per minute for your plan. The response headers include Retry-After,
which specifies how many seconds to wait before retrying. Implement exponential backoff
in your application: wait 1 second, retry, wait 2 seconds on the next failure, wait
4 seconds, and so on up to a maximum of 60 seconds. Do not retry immediately after a
429. Persistent 429 errors during normal usage indicate that you should request a rate
limit increase through the account settings.

A 401 error means the API key in the request is invalid, has been revoked, or does not
have permission for the requested resource. Check that the Authorization header is
formatted as Bearer YOUR_API_KEY with no extra spaces. If your API key was recently
regenerated, ensure the old key has been replaced everywhere it is used. Keys can be
managed in the API Keys section of your account settings.

A 503 error indicates temporary service degradation or a planned maintenance window.
Check the status page at status.example.com for current information. During service
degradation, implement the same exponential backoff strategy used for 429 errors. 503
errors typically resolve within 15 minutes.

## Response Latency

Average response latency for the Standard plan is under 2 seconds for requests up to
1000 tokens. Requests with longer prompts or larger outputs have proportionally higher
latency. Streaming mode reduces the time to first token. For latency-sensitive
applications, use the smallest model that meets your quality requirements.

## Enterprise SLA

Enterprise customers are covered by a 99.9 percent uptime SLA, measured monthly.
Downtime that falls below the SLA threshold is credited back as service credits applied
to the next invoice. SLA credits require a formal support ticket filed within 30 days
of the incident. Credits do not apply to planned maintenance windows announced at least
48 hours in advance.
""",

    "api_models.md": """\
# API Models Reference

## Available Models

The platform provides access to the following models: gpt-4o, gpt-4o-mini,
text-embedding-3-small, and text-embedding-3-large. Each model is designed for specific
use cases and has different performance and cost characteristics.

## Chat Completion Models

gpt-4o is the highest-capability model available. It has a 128,000-token context
window and supports text and image inputs. It produces the strongest results for complex
reasoning, analysis, and generation tasks. The cost is $5.00 per million input tokens
and $15.00 per million output tokens.

gpt-4o-mini is optimized for speed and cost efficiency while retaining strong language
understanding. It has a 128,000-token context window and is recommended for tasks that
are straightforward or latency-sensitive. The cost is $0.15 per million input tokens
and $0.60 per million output tokens. Most developers should start with gpt-4o-mini and
upgrade to gpt-4o only where the quality difference is measurable.

## Embedding Models

text-embedding-3-small produces 1536-dimensional vectors and is the recommended choice
for new projects. It outperforms the previous generation text-embedding-ada-002 model
at a lower cost. The cost is $0.02 per million tokens. Because the model is smaller, it
also has lower latency, making it well-suited for real-time applications that require
embedding at query time.

text-embedding-3-large produces 3072-dimensional vectors and achieves higher retrieval
accuracy on benchmarks. The cost is $0.13 per million tokens, which is more than six
times the cost of text-embedding-3-small. The quality improvement is meaningful but
not always worth the cost increase. Run an offline evaluation on your specific retrieval
task before committing to the larger model.

## Choosing an Embedding Model

For most RAG and semantic search applications, text-embedding-3-small is the right
starting point. Its 1536-dimensional vectors are large enough to capture nuanced
semantics for most domain-specific corpora. Switch to text-embedding-3-large if offline
evaluation shows that retrieval quality is limiting your application's performance.
Important: once you have indexed a corpus with one embedding model, you must re-index
with the other model before switching. Mixing index and query embeddings from different
models produces incorrect similarity scores and silently degrades retrieval quality.

## Rate Limits

Rate limits are applied per API key. The Standard plan allows 3,500 requests per minute
for chat completion and 1,000,000 tokens per minute for embeddings. Contact support to
request a rate limit increase.
""",
}


def main() -> None:
    print("\n=== Step 1: Create knowledge base ===\n")

    os.makedirs(DATA_DIR, exist_ok=True)

    for fname, content in DOCUMENTS.items():
        path = os.path.join(DATA_DIR, fname)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    print(f"Created {len(DOCUMENTS)} documents in {DATA_DIR}/\n")

    col_name  = 26
    col_words = 8
    col_chars = 9
    header = f"{'File':<{col_name}}  {'Words':>{col_words}}  {'Chars':>{col_chars}}"
    print(header)
    print("-" * (col_name + col_words + col_chars + 6))

    total_words = 0
    for fname, content in DOCUMENTS.items():
        words = len(content.split())
        chars = len(content)
        total_words += words
        print(f"{fname:<{col_name}}  {words:>{col_words}}  {chars:>{col_chars}}")

    print("-" * (col_name + col_words + col_chars + 6))
    print(f"{'TOTAL':<{col_name}}  {total_words:>{col_words}}")

    print(
        "\nCommon pitfall: documents without clear headings produce poor chunks."
        "\nThese docs use H2 headers to create natural section boundaries."
    )
    print("\nDone. Run step2_chunking.py next.")


if __name__ == "__main__":
    main()
