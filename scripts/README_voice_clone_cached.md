# Cached no-transcript voice clone

This helper uses `Qwen/Qwen3-TTS-12Hz-1.7B-Base` in `x_vector_only_mode=True`.
It only needs reference audio. The first run creates a reusable voice cache; later
runs reuse the cache and skip reference-audio feature extraction.

## Install with uv and Tsinghua mirror

```bash
cd /root/codex/qwen-TTS/Qwen3-TTS
UV_CACHE_DIR=.uv-cache uv venv --python python3.11
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python -e . \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

Optional, if your GPU supports FlashAttention:

```bash
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python flash-attn --no-build-isolation \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## Run

You can put parameters in [configs/voice_clone_cached.toml](../configs/voice_clone_cached.toml):

```bash
.venv/bin/python scripts/voice_clone_cached.py \
  --config configs/voice_clone_cached.toml
```

Command-line options override the config file:

```bash
.venv/bin/python scripts/voice_clone_cached.py \
  --config configs/voice_clone_cached.toml \
  --text "这次临时换一段文本。" \
  --output outputs/override.wav
```

Or pass all parameters on the command line:

```bash
.venv/bin/python scripts/voice_clone_cached.py \
  --model models/Qwen3-TTS-12Hz-1.7B-Base \
  --ref-audio /path/to/reference.wav \
  --cache caches/my_voice.pt \
  --text "你好，这是用参考音频音色生成的中文语音。" \
  --output outputs/result.wav \
  --language Chinese \
  --device cuda:0
```

When running through Codex, GPU commands need sandbox-external approval. In a
normal terminal, `--device cuda:0` is enough if `torch.cuda.is_available()` is
true.

If `flash-attn` is installed:

```bash
--attn-implementation flash_attention_2
```

`flash-attn` is optional. If the machine has no `nvcc`/`CUDA_HOME`, installation
from source will fail; the script still runs on GPU with the default `eager`
attention implementation, just with lower speed/memory efficiency.

To rebuild the cache:

```bash
--force-rebuild
```

To verify the reference audio duration before creating a cache:

```bash
.venv/bin/python scripts/voice_clone_cached.py \
  --config configs/voice_clone_cached.toml \
  --inspect-ref-audio
```

When a cache is created, the script also stores `ref_audio_metadata` in the
cache payload, including sample rate, sample count, duration in seconds, and
estimated 12.5 Hz tokenizer frame count.

## Smoke test

This does not download the model. It verifies first-run cache creation and
second-run cache reuse with a mock backend.

```bash
UV_CACHE_DIR=.uv-cache uv run --no-sync python scripts/smoke_test_voice_clone_cached.py
```
