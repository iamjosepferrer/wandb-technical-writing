"""
05_compare_checkpoints.py
-------------------------
Evaluates multiple checkpoints from a training run and logs all results
to the same Weights & Biases project so you can compare them visually.

This is useful for deciding which epoch to use for inference:
  - Early epochs may under-fit (voice drifts)
  - Late epochs may over-fit (the model stops generalising to new text)
  - The sweet spot is usually somewhere in the middle

Usage:
  python 05_compare_checkpoints.py \
    --output_dir ./output \
    --speaker_name my_speaker \
    --ref_audio ./data/ref.wav \
    --test_sentences ./test_sentences.txt \
    --wandb_project qwen3-tts-finetuning \
    --epochs 2 4 6 8 10

This calls 04_evaluate.py logic for each epoch checkpoint in sequence
and logs all results under the same W&B project, letting you compare
SECS and WER across epochs in a single chart.
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True,
                        help="Directory containing checkpoint-epoch-N subdirs")
    parser.add_argument("--speaker_name", required=True)
    parser.add_argument("--ref_audio", required=True)
    parser.add_argument("--test_sentences", required=True)
    parser.add_argument("--wandb_project", default="qwen3-tts-finetuning")
    parser.add_argument("--epochs", nargs="+", type=int, required=True,
                        help="Which epoch checkpoints to evaluate, e.g. --epochs 2 4 6 8 10")
    args = parser.parse_args()

    results = []

    for epoch in args.epochs:
        checkpoint_path = os.path.join(args.output_dir, f"checkpoint-epoch-{epoch}")
        if not os.path.isdir(checkpoint_path):
            print(f"Checkpoint not found, skipping: {checkpoint_path}")
            continue

        print(f"\n{'='*50}")
        print(f"Evaluating epoch {epoch}: {checkpoint_path}")
        print(f"{'='*50}")

        cmd = [
            sys.executable, "04_evaluate.py",
            "--model_path", checkpoint_path,
            "--speaker_name", args.speaker_name,
            "--ref_audio", args.ref_audio,
            "--test_sentences", args.test_sentences,
            "--wandb_project", args.wandb_project,
            "--run_name", f"checkpoint-epoch-{epoch}",
        ]

        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            print(f"WARNING: Evaluation failed for epoch {epoch}")
        else:
            results.append(epoch)

    print(f"\nCompleted evaluation for epochs: {results}")
    print(f"Open your Weights & Biases project to compare them:")
    print(f"  https://wandb.ai/your-username/{args.wandb_project}/runs")


if __name__ == "__main__":
    main()