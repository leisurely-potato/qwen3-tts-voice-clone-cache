#!/usr/bin/env python3
"""Cached no-transcript voice cloning for Qwen3-TTS 1.7B Base.

This script uses x-vector-only mode, so it only needs a reference audio file.
The first run builds and saves a reusable voice-clone prompt. Later runs with
the same cache file skip prompt extraction and synthesize directly.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import struct
import sys
import tomllib
import wave
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
CACHE_VERSION = 1


def sha256_file(path: str) -> str | None:
    if path.startswith(("http://", "https://")):
        return None

    p = Path(path)
    if not p.is_file():
        return None

    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_text_arg(value: str) -> str:
    if value.startswith("@"):
        return Path(value[1:]).read_text(encoding="utf-8")
    return value


def ensure_parent(path: str) -> None:
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def load_real_model(args: argparse.Namespace):
    import torch
    from qwen_tts import Qwen3TTSModel

    device = args.device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print(
            "Warning: using CPU. Qwen3-TTS 1.7B inference can be very slow without a CUDA GPU.",
            file=sys.stderr,
        )

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    kwargs: dict[str, Any] = {
        "device_map": device,
        "dtype": dtype_map[args.dtype],
    }
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    print(f"Loading model: {args.model}", file=sys.stderr)
    return Qwen3TTSModel.from_pretrained(args.model, **kwargs)


def inspect_ref_audio(model, args: argparse.Namespace):
    normalized = model._normalize_audio_inputs(args.ref_audio)
    metadata = []
    for i, (wav, sr) in enumerate(normalized):
        samples = int(wav.shape[0])
        duration = samples / float(sr)
        estimated_12hz_frames = duration * 12.5
        item = {
            "index": i,
            "sample_rate": int(sr),
            "samples": samples,
            "duration_seconds": duration,
            "estimated_12hz_frames": estimated_12hz_frames,
        }
        metadata.append(item)
        print(
            "Reference audio "
            f"#{i}: sr={sr}, samples={samples}, duration={duration:.3f}s "
            f"({duration / 60:.2f}min), estimated_12hz_frames={estimated_12hz_frames:.1f}",
            file=sys.stderr,
        )
    return normalized, metadata


def save_real_prompt(cache_path: str, model, args: argparse.Namespace) -> None:
    import torch

    print("Cache not found; extracting voice prompt from reference audio.", file=sys.stderr)
    normalized, ref_audio_metadata = inspect_ref_audio(model, args)
    ref_audio_for_prompt = normalized if len(normalized) > 1 else normalized[0]
    prompt_items = model.create_voice_clone_prompt(
        ref_audio=ref_audio_for_prompt,
        ref_text=None,
        x_vector_only_mode=True,
    )
    payload = {
        "version": CACHE_VERSION,
        "backend": "qwen3-tts",
        "model": args.model,
        "mode": "x_vector_only",
        "ref_audio": args.ref_audio,
        "ref_audio_sha256": sha256_file(args.ref_audio),
        "ref_audio_metadata": ref_audio_metadata,
        "items": [dataclasses.asdict(item) for item in prompt_items],
    }
    ensure_parent(cache_path)
    torch.save(payload, cache_path)
    print(f"Saved voice prompt cache: {cache_path}", file=sys.stderr)


def load_real_prompt(cache_path: str):
    import torch
    from qwen_tts import VoiceClonePromptItem

    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("Invalid cache: expected a dictionary payload.")
    if payload.get("backend") != "qwen3-tts":
        raise ValueError(f"Cache backend is {payload.get('backend')!r}, not 'qwen3-tts'.")
    if payload.get("mode") != "x_vector_only":
        raise ValueError(f"Cache mode is {payload.get('mode')!r}, not 'x_vector_only'.")

    raw_items = payload.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        raise ValueError("Invalid cache: missing prompt items.")

    items = []
    for item in raw_items:
        if not isinstance(item, dict):
            raise ValueError("Invalid cache: prompt item is not a dictionary.")
        ref_spk = item.get("ref_spk_embedding")
        if ref_spk is None:
            raise ValueError("Invalid cache: missing ref_spk_embedding.")
        if not torch.is_tensor(ref_spk):
            ref_spk = torch.tensor(ref_spk)

        ref_code = item.get("ref_code")
        if ref_code is not None and not torch.is_tensor(ref_code):
            ref_code = torch.tensor(ref_code)

        items.append(
            VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=ref_spk,
                x_vector_only_mode=bool(item.get("x_vector_only_mode", True)),
                icl_mode=bool(item.get("icl_mode", False)),
                ref_text=item.get("ref_text"),
            )
        )
    return items


def run_real(args: argparse.Namespace) -> None:
    import soundfile as sf

    model = load_real_model(args)
    if args.inspect_ref_audio:
        inspect_ref_audio(model, args)
        return

    cache_path = str(Path(args.cache).expanduser())

    if args.force_rebuild or not Path(cache_path).exists():
        save_real_prompt(cache_path, model, args)
    else:
        print(f"Using existing voice prompt cache: {cache_path}", file=sys.stderr)

    prompt_items = load_real_prompt(cache_path)

    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "subtalker_dosample": True,
        "subtalker_top_k": args.top_k,
        "subtalker_top_p": args.top_p,
        "subtalker_temperature": args.temperature,
    }
    text = read_text_arg(args.text).strip()
    if not text:
        raise ValueError("Target text is empty.")

    print("Generating audio.", file=sys.stderr)
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=args.language,
        voice_clone_prompt=prompt_items,
        **gen_kwargs,
    )

    ensure_parent(args.output)
    sf.write(args.output, wavs[0], sr)
    print(f"Wrote output audio: {args.output}", file=sys.stderr)


def write_tone_wav(path: str, text: str, sample_rate: int = 24000) -> None:
    ensure_parent(path)
    duration = min(3.0, max(0.5, len(text) / 30.0))
    total = int(sample_rate * duration)
    freq = 330.0 + (len(text) % 20) * 11.0
    with wave.open(path, "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        for i in range(total):
            value = int(12000 * math.sin(2 * math.pi * freq * i / sample_rate))
            f.writeframesraw(struct.pack("<h", value))


def run_mock(args: argparse.Namespace) -> None:
    cache_path = Path(args.cache).expanduser()
    if args.force_rebuild or not cache_path.exists():
        print("Mock cache not found; creating it.", file=sys.stderr)
        ensure_parent(str(cache_path))
        payload = {
            "version": CACHE_VERSION,
            "backend": "mock",
            "model": args.model,
            "mode": "x_vector_only",
            "ref_audio": args.ref_audio,
            "ref_audio_sha256": sha256_file(args.ref_audio),
        }
        cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(f"Using existing mock cache: {cache_path}", file=sys.stderr)

    text = read_text_arg(args.text).strip()
    if not text:
        raise ValueError("Target text is empty.")
    write_tone_wav(args.output, text)
    print(f"Wrote mock output audio: {args.output}", file=sys.stderr)


def load_config(path: str) -> dict[str, Any]:
    config_path = Path(path).expanduser()
    with config_path.open("rb") as f:
        data = tomllib.load(f)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a TOML table.")
    return data


def apply_config_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser) -> argparse.Namespace:
    config_path = getattr(args, "config", None)
    config = load_config(config_path) if config_path else {}
    allowed = {action.dest for action in parser._actions}
    unknown = sorted(k for k in config if k not in allowed)
    if unknown:
        raise ValueError(f"Unknown config option(s): {', '.join(unknown)}")

    defaults = {
        "model": DEFAULT_MODEL,
        "output": "output.wav",
        "language": "Chinese",
        "device": "auto",
        "dtype": "bfloat16",
        "attn_implementation": "eager",
        "force_rebuild": False,
        "max_new_tokens": 2048,
        "top_k": 50,
        "top_p": 1.0,
        "temperature": 0.9,
        "repetition_penalty": 1.05,
        "mock": False,
        "inspect_ref_audio": False,
    }
    cli_values = {k: v for k, v in vars(args).items() if k != "config"}
    merged = {**defaults, **config, **cli_values}
    if config_path:
        merged["config"] = config_path
    args = argparse.Namespace(**merged)

    missing = [
        name
        for name in ("ref_audio", "cache", "text")
        if not getattr(args, name, None)
    ]
    if missing:
        raise ValueError(
            "Missing required option(s): "
            + ", ".join(missing)
            + ". Set them in the config file or pass them on the command line."
        )
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate speech with Qwen3-TTS 1.7B Base using a cached no-transcript "
            "voice prompt from reference audio."
        )
    )
    parser.add_argument("--config", help="TOML config file. Command-line options override config values.")
    parser.add_argument("--ref-audio", default=argparse.SUPPRESS, help="Reference audio path/URL/base64.")
    parser.add_argument("--cache", default=argparse.SUPPRESS, help="Path to save/load the extracted voice prompt cache.")
    parser.add_argument("--text", default=argparse.SUPPRESS, help="Target text, or @path/to/text.txt.")
    parser.add_argument("--output", default=argparse.SUPPRESS, help="Output wav path.")
    parser.add_argument("--language", default=argparse.SUPPRESS, help="Target language, e.g. Chinese, English, Japanese, Auto.")
    parser.add_argument("--model", default=argparse.SUPPRESS, help="Qwen3-TTS Base model id/path.")
    parser.add_argument(
        "--device",
        default=argparse.SUPPRESS,
        help="Device map passed to Qwen3TTSModel.from_pretrained. Use auto, cuda:0, or cpu.",
    )
    parser.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default=argparse.SUPPRESS)
    parser.add_argument("--attn-implementation", default=argparse.SUPPRESS)
    parser.add_argument(
        "--force-rebuild",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Rebuild cache even if it already exists.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--top-k", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--top-p", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--temperature", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--repetition-penalty", type=float, default=argparse.SUPPRESS)
    parser.add_argument(
        "--mock",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Run a lightweight local smoke test backend instead of loading Qwen3-TTS.",
    )
    parser.add_argument(
        "--inspect-ref-audio",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Load reference audio, print duration/sample metadata, then exit without generating.",
    )
    return parser


def main() -> int:
    try:
        parser = build_parser()
        args = apply_config_defaults(parser.parse_args(), parser)
        if args.mock:
            run_mock(args)
        else:
            run_real(args)
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
