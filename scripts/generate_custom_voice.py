import argparse
from pathlib import Path

import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel


def _dtype(value: str):
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float16":
        return torch.float16
    if value == "float32":
        return torch.float32
    if value == "auto":
        return "auto"
    raise ValueError(f"Unsupported dtype: {value}")


def main():
    parser = argparse.ArgumentParser(description="Generate audio from a Qwen3-TTS CustomVoice checkpoint.")
    parser.add_argument("--model", required=True, help="CustomVoice checkpoint path.")
    parser.add_argument("--text", required=True, help="Text to synthesize.")
    parser.add_argument("--output", required=True, help="Output WAV path.")
    parser.add_argument("--speaker", default="m3", help="Speaker name stored in the checkpoint config.")
    parser.add_argument("--language", default="Auto", help="Language hint, such as Chinese, English, Japanese, or Auto.")
    parser.add_argument("--instruct", default="", help="Optional speaking style instruction.")
    parser.add_argument("--device", default="cuda:0", help="Device map, for example cuda:0 or cpu.")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32", "auto"])
    parser.add_argument("--attn_implementation", default="eager")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    args = parser.parse_args()

    tts = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
    )

    wavs, sample_rate = tts.generate_custom_voice(
        text=args.text,
        language=args.language,
        speaker=args.speaker,
        instruct=args.instruct,
        max_new_tokens=args.max_new_tokens,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, wavs[0], sample_rate)
    print(f"Saved {output_path} ({sample_rate} Hz)")


if __name__ == "__main__":
    main()
