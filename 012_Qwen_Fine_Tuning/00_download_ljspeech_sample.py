"""
00_download_ljspeech_sample.py
------------------------------
Downloads the LJSpeech archive directly from keithito.com and extracts
only the first --num_clips WAV files, leaving the rest untouched.

Why not Hugging Face?
The keithito/lj_speech dataset on HF uses a legacy loading script that
breaks with datasets>=4.0, and even with older versions the script tries
to extract the full .tar.bz2 in streaming mode, which is not supported.
Downloading the archive directly avoids all of that.

The archive is 2.6 GB and must be downloaded in full (bz2 is a sequential
format so partial downloads are not possible). It is cached to
./data/cache/ so re-running the script skips the download entirely.

Only the first --num_clips WAV files are extracted to disk. The rest of
the archive is read and discarded, so the full 24-hour dataset is never
written to disk. 200 clips gives ~15 minutes of audio, enough to run
the full pipeline end to end.

Usage:
  python 00_download_ljspeech_sample.py
  python 00_download_ljspeech_sample.py --num_clips 500

Requirements:
  pip install soundfile tqdm numpy
  (all already in requirements.txt)
"""

import argparse
import csv
import os
import shutil
import tarfile
import urllib.request

import soundfile as sf
from tqdm import tqdm


LJSPEECH_URL = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"


# ------------------------------------------------------------------ #
# Download
# ------------------------------------------------------------------ #

def download_archive(url: str, dst: str) -> None:
    """Download the LJSpeech archive with a progress bar.
    Skips download if the file already exists in cache."""
    if os.path.exists(dst):
        size_mb = os.path.getsize(dst) / 1_000_000
        print(f"Archive already cached ({size_mb:.0f} MB): {dst}")
        print("Skipping download.\n")
        return

    print(f"Downloading LJSpeech archive (~2.6 GB) to {dst}")
    print("This only happens once. Subsequent runs use the cached file.\n")

    with tqdm(
        unit="B", unit_scale=True, unit_divisor=1024,
        miniters=1, desc="LJSpeech-1.1.tar.bz2"
    ) as progress:
        def _hook(blocks_transferred: int, block_size: int, total_size: int) -> None:
            if total_size > 0:
                progress.total = total_size
            progress.n = blocks_transferred * block_size
            progress.refresh()

        urllib.request.urlretrieve(url, dst, reporthook=_hook)

    print("\nDownload complete.")


# ------------------------------------------------------------------ #
# Selective extraction
# ------------------------------------------------------------------ #

def extract_sample(
    archive_path: str,
    extract_dir: str,
    num_clips: int,
) -> tuple[str, str]:
    """
    Extract metadata.csv and the first num_clips WAV files from the archive.
    Stops iterating as soon as both are collected, so the rest of the
    24-hour dataset is never written to disk.

    Returns (wavs_dir, metadata_csv_path).
    """
    root_dir = os.path.join(extract_dir, "LJSpeech-1.1")
    wavs_dir = os.path.join(root_dir, "wavs")
    metadata_csv_path = os.path.join(root_dir, "metadata.csv")

    os.makedirs(wavs_dir, exist_ok=True)

    # Skip if we already have enough clips from a previous run
    if os.path.isfile(metadata_csv_path):
        existing = len([f for f in os.listdir(wavs_dir) if f.endswith(".wav")])
        if existing >= num_clips:
            print(f"Already extracted {existing} clips. Skipping extraction.")
            return wavs_dir, metadata_csv_path

    print(f"Extracting metadata.csv and first {num_clips} clips from archive...")
    print("(The rest of the archive is skipped)\n")

    wav_count = 0
    metadata_done = False

    with tarfile.open(archive_path, "r:bz2") as tar:
        with tqdm(desc="Extracting", unit=" files") as progress:
            for member in tar:
                name = member.name

                # Extract metadata.csv (comes near the top of the archive)
                if name.endswith("metadata.csv") and not metadata_done:
                    tar.extract(member, path=extract_dir, set_attrs=False)
                    metadata_done = True
                    progress.update(1)

                # Extract WAV files until we have enough
                elif name.endswith(".wav") and wav_count < num_clips:
                    tar.extract(member, path=extract_dir, set_attrs=False)
                    wav_count += 1
                    progress.update(1)

                # Stop as soon as we have what we need
                if metadata_done and wav_count >= num_clips:
                    break

    print(f"\nExtracted: metadata.csv + {wav_count} WAV files")
    return wavs_dir, metadata_csv_path


# ------------------------------------------------------------------ #
# Reference clip selection
# ------------------------------------------------------------------ #

def find_reference_clip(
    records: list[dict],
    min_sec: float = 8.0,
    max_sec: float = 12.0,
) -> dict:
    """Return the first clip whose duration falls in [min_sec, max_sec].
    Falls back to the clip closest to 9 seconds if none qualifies."""
    for record in records:
        if min_sec <= record["duration"] <= max_sec:
            return record
    return min(records, key=lambda r: abs(r["duration"] - 9.0))


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Download a sample of LJSpeech for Qwen3-TTS fine-tuning."
    )
    parser.add_argument(
        "--num_clips", type=int, default=200,
        help="Number of clips to extract (default: 200, ~15 min of audio)"
    )
    parser.add_argument(
        "--output_dir", default="./data/raw",
        help="Where to write the sampled WAV files"
    )
    parser.add_argument(
        "--metadata_path", default="./data/ljspeech_metadata.csv",
        help="Where to write the transcript CSV for Step 2"
    )
    parser.add_argument(
        "--ref_audio_path", default="./data/ref.wav",
        help="Where to save the reference speaker clip"
    )
    parser.add_argument(
        "--cache_dir", default="./data/cache",
        help="Where to cache the downloaded archive"
    )
    args = parser.parse_args()

    # Create necessary directories
    for directory in [args.output_dir, args.cache_dir]:
        os.makedirs(directory, exist_ok=True)
    metadata_dir = os.path.dirname(args.metadata_path)
    if metadata_dir:
        os.makedirs(metadata_dir, exist_ok=True)

    # Step 1: Download archive to cache (skipped on subsequent runs)
    archive_path = os.path.join(args.cache_dir, "LJSpeech-1.1.tar.bz2")
    download_archive(LJSPEECH_URL, archive_path)

    # Step 2: Extract only what we need from the archive
    wavs_dir, metadata_csv_path = extract_sample(
        archive_path, args.cache_dir, args.num_clips
    )

    # Step 3: Read transcript lookup from metadata.csv
    # LJSpeech metadata.csv is pipe-delimited with no header row:
    #   clip_id | raw_text | normalized_text
    # We use normalized_text: numbers and abbreviations are already
    # spelled out in full, which produces cleaner tokenizer alignments.
    id_to_text: dict[str, str] = {}
    with open(metadata_csv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 3:
                id_to_text[parts[0]] = parts[2].strip()
            elif len(parts) == 2:
                id_to_text[parts[0]] = parts[1].strip()

    # Step 4: Copy extracted clips to output_dir and build metadata records
    print(f"\nCopying clips to {args.output_dir}...")

    extracted_wavs = sorted(
        f for f in os.listdir(wavs_dir) if f.endswith(".wav")
    )[:args.num_clips]

    if not extracted_wavs:
        raise RuntimeError(
            f"No WAV files found in {wavs_dir}. "
            "Extraction may have failed — check the archive at "
            f"{archive_path} and try deleting it to force a re-download."
        )

    records: list[dict] = []
    total_duration = 0.0

    for fname in tqdm(extracted_wavs, desc="Processing clips"):
        clip_id = os.path.splitext(fname)[0]
        src_path = os.path.join(wavs_dir, fname)
        dst_path = os.path.join(args.output_dir, fname)

        # sf.info reads only the WAV header — much faster than loading the audio
        info = sf.info(src_path)
        duration = info.frames / info.samplerate
        total_duration += duration

        shutil.copy(src_path, dst_path)

        records.append({
            "id": clip_id,
            "wav_path": dst_path,
            "text": id_to_text.get(clip_id, ""),
            "duration": round(duration, 2),
        })

    print(f"\nSaved {len(records)} files.")
    print(f"Total audio: {total_duration:.1f}s ({total_duration / 60:.1f} min)")

    # Step 5: Write metadata CSV for 02_build_jsonl_from_ljspeech.py
    with open(args.metadata_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "wav_path", "text", "duration"]
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"Metadata written to {args.metadata_path}")

    # Step 6: Select and copy reference clip
    ref = find_reference_clip(records)
    shutil.copy(ref["wav_path"], args.ref_audio_path)
    print(
        f"Reference clip: {ref['id']} ({ref['duration']:.1f}s)"
        f" -> {args.ref_audio_path}"
    )

    print("\nDone. Next step:")
    print(
        f"  python 01_prepare_audio.py"
        f" --input_dir {args.output_dir}"
        f" --output_dir ./data/clean"
    )


if __name__ == "__main__":
    main()
