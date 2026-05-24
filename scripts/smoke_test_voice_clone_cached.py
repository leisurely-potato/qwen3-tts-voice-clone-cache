#!/usr/bin/env python3
"""Smoke test for scripts/voice_clone_cached.py without downloading Qwen3-TTS."""

from __future__ import annotations

import math
import os
import random
import struct
import subprocess
import sys
import tempfile
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "scripts" / "voice_clone_cached.py"


def write_random_reference(path: Path) -> None:
    rng = random.Random(20260524)
    sample_rate = 24000
    seconds = 1.0
    freq = rng.uniform(180.0, 320.0)
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        for i in range(int(sample_rate * seconds)):
            noise = rng.uniform(-0.03, 0.03)
            sample = math.sin(2 * math.pi * freq * i / sample_rate) * 0.4 + noise
            value = max(-32768, min(32767, int(sample * 32767)))
            f.writeframesraw(struct.pack("<h", value))


def run_cli(ref: Path, cache: Path, output: Path, text: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(CLI),
            "--mock",
            "--ref-audio",
            str(ref),
            "--cache",
            str(cache),
            "--text",
            text,
            "--output",
            str(output),
            "--language",
            "Chinese",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=False,
    )


def assert_wav(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise AssertionError(f"Missing output wav: {path}")
    with wave.open(str(path), "rb") as f:
        if f.getnchannels() != 1:
            raise AssertionError("Expected mono wav.")
        if f.getframerate() != 24000:
            raise AssertionError("Expected 24 kHz wav.")
        if f.getnframes() <= 0:
            raise AssertionError("Expected at least one audio frame.")


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="qwen3_tts_cached_smoke_") as d:
        work = Path(d)
        ref = work / "random_ref.wav"
        cache = work / "voice_cache.json"
        out1 = work / "out_first.wav"
        out2 = work / "out_second.wav"
        write_random_reference(ref)

        first = run_cli(ref, cache, out1, "第一次生成，应该创建缓存。")
        if first.returncode != 0:
            print(first.stderr, file=sys.stderr)
            return first.returncode
        if "creating it" not in first.stderr:
            print(first.stderr, file=sys.stderr)
            raise AssertionError("First run did not create the cache.")
        assert_wav(out1)
        first_mtime_ns = os.stat(cache).st_mtime_ns

        second = run_cli(ref, cache, out2, "第二次生成，应该直接复用缓存。")
        if second.returncode != 0:
            print(second.stderr, file=sys.stderr)
            return second.returncode
        if "Using existing mock cache" not in second.stderr:
            print(second.stderr, file=sys.stderr)
            raise AssertionError("Second run did not reuse the cache.")
        assert_wav(out2)
        second_mtime_ns = os.stat(cache).st_mtime_ns
        if first_mtime_ns != second_mtime_ns:
            raise AssertionError("Cache was unexpectedly modified on the second run.")

        print("smoke test passed")
        print(f"reference={ref}")
        print(f"cache={cache}")
        print(f"outputs={out1}, {out2}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
