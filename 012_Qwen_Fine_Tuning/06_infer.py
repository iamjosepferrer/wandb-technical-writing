"""
06_infer.py
-----------
Run inference with your fine-tuned Qwen3-TTS model.

Usage:
  python 06_infer.py \
    --model_path ./output/checkpoint-epoch-5 \
    --speaker_name my_speaker \
    --text "Hello, this is a test of my custom voice." \
    --output output.wav
"""

import argparse
import torch
import soundfile as sf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--speaker_name", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--output", default="output.wav")
    parser.add_argument("--language", default="en")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    from qwen_tts import Qwen3TTSModel
    print(f"Loading model from {args.model_path}...")
    tts = Qwen3TTSModel.from_pretrained(
        args.model_path,
        device_map=device,
        dtype=torch.bfloat16,
    )

    print(f"Generating: {args.text}")
    wavs, sr = tts.generate_custom_voice(
        text=args.text,
        speaker=args.speaker_name,
        language=args.language,
    )

    sf.write(args.output, wavs[0], sr)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()