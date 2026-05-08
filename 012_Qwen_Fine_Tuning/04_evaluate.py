"""
04_evaluate.py
--------------
Evaluates a fine-tuned Qwen3-TTS model by measuring:

  1. Speaker Embedding Cosine Similarity (SECS)
     Uses Resemblyzer d-vector embeddings. A score >0.85 indicates
     good voice preservation. The original Qwen3-TTS paper reports
     0.89 for the 1.7B base model on zero-shot cloning.

  2. Word Error Rate (WER)
     Transcribes generated audio with Whisper and compares against
     the expected text. Low WER means the model is pronouncing
     words correctly.

All results are logged to Weights & Biases so you can compare
checkpoints side-by-side in the W&B UI.

Usage:
  python 04_evaluate.py \
    --model_path ./output/checkpoint-epoch-5 \
    --speaker_name my_speaker \
    --ref_audio ./data/ref.wav \
    --test_sentences ./test_sentences.txt \
    --wandb_project qwen3-tts-finetuning \
    --run_name checkpoint-epoch-5

test_sentences.txt: one sentence per line, these are the texts you'll
synthesise and evaluate. Keep them diverse: short sentences, long ones,
sentences with numbers and names.
"""

import argparse
import os
import tempfile

import numpy as np
import soundfile as sf
import torch
import wandb
import whisper
from jiwer import wer
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
from tqdm import tqdm


def load_test_sentences(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
    if not sentences:
        raise ValueError(f"No sentences found in {path}")
    return sentences


def compute_speaker_similarity(
    encoder: VoiceEncoder,
    ref_audio_path: str,
    gen_audio_path: str,
) -> float:
    """
    Compute cosine similarity between reference and generated audio
    using Resemblyzer d-vector embeddings.
    """
    ref_wav = preprocess_wav(ref_audio_path)
    gen_wav = preprocess_wav(gen_audio_path)

    ref_embed = encoder.embed_utterance(ref_wav)
    gen_embed = encoder.embed_utterance(gen_wav)

    # Cosine similarity
    similarity = float(np.dot(ref_embed, gen_embed) / (
        np.linalg.norm(ref_embed) * np.linalg.norm(gen_embed)
    ))
    return similarity


def compute_wer(whisper_model, audio_path: str, expected_text: str) -> float:
    """Transcribe generated audio and compare to expected text."""
    result = whisper_model.transcribe(audio_path, verbose=False)
    hypothesis = result["text"].strip().lower()
    reference = expected_text.strip().lower()
    return wer(reference, hypothesis)


def run_evaluation(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- W&B ----
    run = wandb.init(
        project=args.wandb_project,
        name=args.run_name or f"eval_{os.path.basename(args.model_path)}",
        job_type="evaluation",
        config={
            "model_path": args.model_path,
            "speaker_name": args.speaker_name,
            "ref_audio": args.ref_audio,
        },
    )

    # ---- Load models ----
    print("Loading Qwen3-TTS model...")
    from qwen_tts import Qwen3TTSModel
    tts = Qwen3TTSModel.from_pretrained(
        args.model_path,
        device_map=device,
        dtype=torch.bfloat16,
    )

    print("Loading Resemblyzer speaker encoder...")
    encoder = VoiceEncoder()

    print("Loading Whisper for WER...")
    whisper_model = whisper.load_model("medium")

    # ---- Load test sentences ----
    sentences = load_test_sentences(args.test_sentences)
    print(f"Evaluating on {len(sentences)} sentences...")

    # ---- W&B Table to log per-sentence results ----
    eval_table = wandb.Table(
        columns=["sentence", "secs", "wer", "audio"]
    )

    secs_scores = []
    wer_scores = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, sentence in enumerate(tqdm(sentences)):
            # Generate speech
            try:
                wavs, sr = tts.generate_custom_voice(
                    text=sentence,
                    speaker=args.speaker_name,
                )
            except Exception as e:
                print(f"Generation failed for sentence {i}: {e}")
                continue

            gen_path = os.path.join(tmpdir, f"gen_{i:04d}.wav")
            sf.write(gen_path, wavs[0], sr)

            # Speaker similarity
            secs = compute_speaker_similarity(encoder, args.ref_audio, gen_path)
            secs_scores.append(secs)

            # WER
            word_error_rate = compute_wer(whisper_model, gen_path, sentence)
            wer_scores.append(word_error_rate)

            # Add row to W&B table
            eval_table.add_data(
                sentence,
                round(secs, 4),
                round(word_error_rate, 4),
                wandb.Audio(gen_path, sample_rate=sr, caption=sentence),
            )

            # Also log per-sentence metrics as scalars so you can see trend
            wandb.log({
                "eval/secs": secs,
                "eval/wer": word_error_rate,
                "eval/sentence_idx": i,
            })

    # ---- Aggregate summary ----
    mean_secs = float(np.mean(secs_scores)) if secs_scores else 0.0
    mean_wer = float(np.mean(wer_scores)) if wer_scores else 0.0

    wandb.run.summary["eval/mean_secs"] = round(mean_secs, 4)
    wandb.run.summary["eval/mean_wer"] = round(mean_wer, 4)
    wandb.run.summary["eval/num_sentences"] = len(secs_scores)

    # Log the full results table
    wandb.log({"eval/results_table": eval_table})

    print(f"\n--- Evaluation Results ---")
    print(f"  Mean SECS : {mean_secs:.4f}  (target: >0.85)")
    print(f"  Mean WER  : {mean_wer:.4f}  (target: <0.05)")
    print(f"  Sentences : {len(secs_scores)}")
    print(f"\nResults logged to: {run.url}")

    run.finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True,
                        help="Path to fine-tuned model checkpoint directory")
    parser.add_argument("--speaker_name", required=True,
                        help="Speaker name used during training")
    parser.add_argument("--ref_audio", required=True,
                        help="Reference audio of the target speaker (same one used during training)")
    parser.add_argument("--test_sentences", required=True,
                        help="Text file with one sentence per line to synthesise and evaluate")
    parser.add_argument("--wandb_project", default="qwen3-tts-finetuning")
    parser.add_argument("--run_name", default=None,
                        help="W&B run name (defaults to model checkpoint directory name)")
    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if not os.path.isfile(args.ref_audio):
        raise FileNotFoundError(f"ref_audio not found: {args.ref_audio}")

    run_evaluation(args)


if __name__ == "__main__":
    main()