"""
03_train_with_wandb.py
----------------------
Fine-tunes Qwen3-TTS by calling the official sft_12hz.py as a subprocess
and logging all training metrics to Weights & Biases.

Why subprocess instead of direct model loading?
  AutoModelForCausalLM does not recognise the qwen3_tts architecture
  because it is defined inside the qwen_tts package, not in the
  transformers registry. sft_12hz.py loads the model through
  Qwen3TTSModel.from_pretrained() internally and always works.
  This wrapper captures its output line by line and logs to W&B.

Must be run from /content/Qwen3-TTS/finetuning/ (same dir as sft_12hz.py).

Usage:
  python 03_train_with_wandb.py \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --output_model_path ./output \
    --train_jsonl /content/train_with_codes.jsonl \
    --speaker_name ljspeech_sample \
    --batch_size 1 \
    --lr 2e-5 \
    --num_epochs 4 \
    --wandb_project qwen3-tts-ljspeech
"""

import argparse
import os
import re
import subprocess
import sys

import wandb


def parse_loss_line(line: str):
    """
    Parse per-step loss lines produced by sft_12hz.py.
    Format: "Epoch 2 | Step 30 | Loss: 5.9758"
    Returns (epoch, loss) or None.
    """
    m = re.search(r"Epoch\s+(\d+)\s+\|\s+Step\s+\d+\s+\|\s+Loss:\s+([\d.]+)", line)
    if m:
        return int(m.group(1)), float(m.group(2))
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        help="Must be 1.7B-Base — sft_12hz.py has a hard-coded hidden "
                             "dimension of 2048 that only matches the 1.7B model.")
    parser.add_argument("--output_model_path", default="./output")
    parser.add_argument("--train_jsonl", required=True)
    parser.add_argument("--speaker_name", required=True,
                        help="Speaker name used at inference time")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=4)
    parser.add_argument("--wandb_project", default="qwen3-tts-finetuning")
    args = parser.parse_args()

    if not os.path.isfile("sft_12hz.py"):
        raise RuntimeError(
            "sft_12hz.py not found. "
            "Run this script from /content/Qwen3-TTS/finetuning/"
        )

    # ---- W&B init ----
    run = wandb.init(
        project=args.wandb_project,
        name=f"{args.speaker_name}_lr{args.lr}_bs{args.batch_size}",
        config={
            "model": args.init_model_path,
            "speaker": args.speaker_name,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "num_epochs": args.num_epochs,
            "train_jsonl": args.train_jsonl,
        },
    )

    print(f"W&B run: {run.url}\n")

    # ---- Build sft_12hz.py command ----
    cmd = [
        sys.executable, "sft_12hz.py",
        "--init_model_path", args.init_model_path,
        "--output_model_path", args.output_model_path,
        "--train_jsonl", args.train_jsonl,
        "--speaker_name", args.speaker_name,
        "--batch_size", str(args.batch_size),
        "--lr", str(args.lr),
        "--num_epochs", str(args.num_epochs),
    ]

    print("Starting training via sft_12hz.py ...")

    # ---- Run and stream output, parsing loss for W&B ----
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # merge stderr so tqdm bars appear
        text=True,
        bufsize=1,
        env=env,
    )

    global_step = 0
    last_epoch = -1

    for line in process.stdout:
        print(line, end="", flush=True)

        result = parse_loss_line(line)
        if result is not None:
            epoch, loss = result
            wandb.log({
                "train/loss": loss,
                "train/epoch": epoch,
                "train/global_step": global_step,
            }, step=global_step)
            global_step += 1
            last_epoch = epoch

    process.wait()

    if process.returncode != 0:
        print(f"\nERROR: sft_12hz.py exited with code {process.returncode}.")
        print("Check the output above for the error message.")
    else:
        epochs_done = last_epoch + 1 if last_epoch >= 0 else 0
        print(f"\nTraining complete. {epochs_done} epochs finished.")
        wandb.run.summary["total_epochs_completed"] = epochs_done

    run.finish()
    print(f"W&B run finished: {run.url}")


if __name__ == "__main__":
    main()
