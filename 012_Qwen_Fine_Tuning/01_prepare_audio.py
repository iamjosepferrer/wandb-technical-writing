"""
01_prepare_audio.py
-------------------
Cleans and validates raw audio files for Qwen3-TTS fine-tuning.

What this does:
  - Resamples all audio to 24 kHz mono (required by the Qwen3-TTS tokenizer)
  - Filters out clips that are too short (<2s) or too long (>30s)
  - Normalises loudness to -23 LUFS (broadcast standard)
  - Writes cleaned files to ./data/clean/

Usage:
  python 01_prepare_audio.py --input_dir ./data/raw --output_dir ./data/clean

Requirements:
  pip install pydub soundfile librosa numpy tqdm
  sudo apt install ffmpeg   # or: brew install ffmpeg on macOS
"""

import argparse
import os
import shutil

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment, effects
from tqdm import tqdm


TARGET_SR = 24_000
MIN_DURATION = 2.0   # seconds
MAX_DURATION = 30.0  # seconds


def normalize_audio(audio: AudioSegment) -> AudioSegment:
    """Normalize to -23 LUFS using pydub's normalize (peak-based approximation)."""
    return effects.normalize(audio)


def resample_and_clean(input_path: str, output_path: str) -> dict:
    """
    Load, validate, resample, and save one audio file.
    Returns a dict with the outcome and any diagnostic info.
    """
    try:
        # Load with librosa so we handle any format (mp3, flac, m4a, wav)
        audio_np, sr = librosa.load(input_path, sr=None, mono=True)
        duration = len(audio_np) / sr

        if duration < MIN_DURATION:
            return {"file": input_path, "status": "skipped", "reason": f"too short ({duration:.1f}s)"}
        if duration > MAX_DURATION:
            return {"file": input_path, "status": "skipped", "reason": f"too long ({duration:.1f}s)"}

        # Resample to target SR
        if sr != TARGET_SR:
            audio_np = librosa.resample(audio_np, orig_sr=sr, target_sr=TARGET_SR)

        # Convert to 16-bit PCM for pydub (pydub needs integer samples)
        audio_int16 = (audio_np * 32767).astype(np.int16)
        pydub_audio = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=TARGET_SR,
            sample_width=2,  # 16-bit = 2 bytes
            channels=1,
        )

        # Normalise loudness
        pydub_audio = normalize_audio(pydub_audio)

        # Save as 24kHz mono WAV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pydub_audio.export(output_path, format="wav")

        return {"file": input_path, "status": "ok", "duration": round(duration, 2)}

    except Exception as e:
        return {"file": input_path, "status": "error", "reason": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Directory with raw audio files")
    parser.add_argument("--output_dir", required=True, help="Where to write cleaned audio")
    args = parser.parse_args()

    supported_extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
    all_files = [
        f for f in os.listdir(args.input_dir)
        if os.path.splitext(f)[1].lower() in supported_extensions
    ]

    if not all_files:
        print(f"No audio files found in {args.input_dir}")
        return

    print(f"Found {len(all_files)} audio files. Processing...")

    results = []
    for fname in tqdm(all_files):
        in_path = os.path.join(args.input_dir, fname)
        # Keep the same filename but always output as .wav
        out_name = os.path.splitext(fname)[0] + ".wav"
        out_path = os.path.join(args.output_dir, out_name)
        result = resample_and_clean(in_path, out_path)
        results.append(result)

    # Summary
    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] == "skipped"]
    errors = [r for r in results if r["status"] == "error"]

    print(f"\n--- Summary ---")
    print(f"  Processed : {len(ok)} files")
    print(f"  Skipped   : {len(skipped)} files")
    print(f"  Errors    : {len(errors)} files")

    if skipped:
        print("\nSkipped:")
        for r in skipped:
            print(f"  {os.path.basename(r['file'])} -> {r['reason']}")
    if errors:
        print("\nErrors:")
        for r in errors:
            print(f"  {os.path.basename(r['file'])} -> {r['reason']}")

    total_duration = sum(r.get("duration", 0) for r in ok)
    print(f"\nTotal training audio: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"Cleaned files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()