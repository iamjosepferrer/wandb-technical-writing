"""
02_build_jsonl_from_ljspeech.py
--------------------------------
Builds train_raw.jsonl from the LJSpeech pre-existing transcripts.

When you record your own audio you need Whisper to transcribe it
(that's what 02_transcribe.py is for). LJSpeech ships with transcripts
already in metadata.csv, so this script skips Whisper entirely and reads
the text directly from the CSV that 00_download_ljspeech_sample.py wrote.

The output format is the same JSONL that the rest of the pipeline expects:
  {"audio": "path/to/clip.wav", "text": "...", "ref_audio": "path/to/ref.wav"}

Usage:
  python 02_build_jsonl_from_ljspeech.py \\
      --clean_audio_dir ./data/clean \\
      --metadata_csv ./data/ljspeech_metadata.csv \\
      --ref_audio ./data/ref.wav \\
      --output_jsonl train_raw.jsonl

Requirements:
  No additional packages beyond what is already in requirements.txt.
"""

import argparse
import csv
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clean_audio_dir", default="./data/clean",
        help="Directory of cleaned WAV files produced by 01_prepare_audio.py"
    )
    parser.add_argument(
        "--metadata_csv", default="./data/ljspeech_metadata.csv",
        help="Metadata CSV written by 00_download_ljspeech_sample.py"
    )
    parser.add_argument(
        "--ref_audio", default="./data/ref.wav",
        help="Reference speaker clip (same one used in 00_download_ljspeech_sample.py)"
    )
    parser.add_argument(
        "--output_jsonl", default="train_raw.jsonl",
        help="Output JSONL file for the tokenization step"
    )
    args = parser.parse_args()

    if not os.path.isfile(args.metadata_csv):
        raise FileNotFoundError(
            f"Metadata CSV not found: {args.metadata_csv}\n"
            "Run 00_download_ljspeech_sample.py first."
        )
    if not os.path.isdir(args.clean_audio_dir):
        raise FileNotFoundError(
            f"Clean audio directory not found: {args.clean_audio_dir}\n"
            "Run 01_prepare_audio.py first."
        )
    if not os.path.isfile(args.ref_audio):
        raise FileNotFoundError(
            f"Reference audio not found: {args.ref_audio}\n"
            "Run 00_download_ljspeech_sample.py first."
        )

    # Read the metadata CSV
    with open(args.metadata_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        metadata_rows = list(reader)

    # Build a lookup from clip ID to transcript
    id_to_text = {row["id"]: row["text"] for row in metadata_rows}

    # Scan the clean audio directory. 01_prepare_audio.py preserves the
    # original filename stem, so LJ001-0001.wav stays LJ001-0001.wav.
    clean_files = sorted(
        f for f in os.listdir(args.clean_audio_dir) if f.endswith(".wav")
    )

    if not clean_files:
        raise RuntimeError(
            f"No WAV files found in {args.clean_audio_dir}. "
            "Run 01_prepare_audio.py first."
        )

    records = []
    skipped = []

    for fname in clean_files:
        clip_id = os.path.splitext(fname)[0]   # e.g. "LJ001-0001"
        audio_path = os.path.join(args.clean_audio_dir, fname)

        text = id_to_text.get(clip_id)
        if text is None:
            skipped.append(fname)
            continue

        if not text.strip():
            skipped.append(fname)
            continue

        records.append({
            "audio": audio_path,
            "text": text.strip(),
            "ref_audio": args.ref_audio,
        })

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Written {len(records)} records to {args.output_jsonl}")

    if skipped:
        print(f"Skipped {len(skipped)} files (no matching transcript found):")
        for fname in skipped[:10]:
            print(f"  {fname}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    # Sanity-check: print 3 sample records
    print("\nSample records:")
    for record in records[:3]:
        short = record["text"][:80] + "..." if len(record["text"]) > 80 else record["text"]
        print(f"  {os.path.basename(record['audio'])}: {short}")

    print(f"\nNext step:")
    print(f"  cd ../Qwen3-TTS/finetuning && python prepare_data.py \\")
    print(f"      --device cuda:0 \\")
    print(f"      --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \\")
    print(f"      --input_jsonl ../../qwen3-tts-finetuning/train_raw.jsonl \\")
    print(f"      --output_jsonl ../../qwen3-tts-finetuning/train_with_codes.jsonl")


if __name__ == "__main__":
    main()
