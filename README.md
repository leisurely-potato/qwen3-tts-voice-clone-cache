# Qwen3-TTS Voice Clone Cache

[中文文档](README.zh-CN.md)

This project is built on top of the upstream open-source model [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by QwenLM. It is a workflow wrapper for cached no-transcript voice cloning and does not provide the original model weights in this repository.

Cached no-transcript voice cloning workflow for Qwen3-TTS 1.7B. This repository wraps the Qwen3-TTS Base model with a practical command-line workflow that extracts a reusable speaker prompt from reference audio once, then reuses that cache for later text-to-speech generation.

## Overview

Qwen3-TTS supports voice cloning from reference audio. For cases where no transcript is available, this project uses `x_vector_only_mode=True`: only the speaker embedding is used, so the workflow needs a reference audio file but no reference text.

The first run creates a cache file from the reference audio. Later runs load that cache and skip reference-audio feature extraction.

## Features

- No-transcript voice cloning with Qwen3-TTS 1.7B Base
- Reusable speaker prompt cache
- TOML configuration file support
- Command-line overrides for config values
- GPU inference support with `cuda:0`
- Reference-audio inspection for duration, sample count, sample rate, and estimated 12.5 Hz frame count
- Local smoke test mode that does not load the model

## Repository Layout

```text
configs/voice_clone_cached.toml       Example configuration
scripts/voice_clone_cached.py         Main CLI
scripts/generate_custom_voice.py      Generate audio from a fine-tuned CustomVoice checkpoint
scripts/resample_dataset_24k.py       Convert a JSONL dataset audio set to 24 kHz WAV
scripts/smoke_test_voice_clone_cached.py
scripts/README_voice_clone_cached.md  Detailed script notes
```

Local model weights, voice caches, generated audio, and sample audio files are intentionally ignored by Git:

```text
models/
data/
caches/
outputs/
samples/
```

## Requirements

- Python 3.11+
- NVIDIA GPU recommended for the 1.7B model
- Qwen3-TTS model weights downloaded locally or accessible from ModelScope/Hugging Face
- `uv` for environment management

`flash-attn` is optional. The script works with the default `eager` attention implementation. Installing `flash-attn` requires a CUDA Toolkit development environment with `nvcc` and `CUDA_HOME`.

## Installation

Use a local virtual environment:

```bash
cd /root/codex/qwen-TTS/Qwen3-TTS
UV_CACHE_DIR=.uv-cache uv venv --python python3.11
```

Install Python dependencies. In mainland China, prefer the Tsinghua PyPI mirror:

```bash
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python -e . \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

Optional FlashAttention installation:

```bash
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python flash-attn --no-build-isolation \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## Download Model Weights

ModelScope is recommended in mainland China:

```bash
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python modelscope \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple

.venv/bin/modelscope download \
  --model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --local_dir models/Qwen3-TTS-12Hz-1.7B-Base

.venv/bin/modelscope download \
  --model Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --local_dir models/Qwen3-TTS-Tokenizer-12Hz
```

Hugging Face mirror alternative:

```bash
HF_ENDPOINT=https://hf-mirror.com \
.venv/bin/huggingface-cli download \
  Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --local-dir models/Qwen3-TTS-12Hz-1.7B-Base

HF_ENDPOINT=https://hf-mirror.com \
.venv/bin/huggingface-cli download \
  Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --local-dir models/Qwen3-TTS-Tokenizer-12Hz
```

## Configure

Edit:

```text
configs/voice_clone_cached.toml
```

Example:

```toml
model = "models/Qwen3-TTS-12Hz-1.7B-Base"
ref_audio = "samples/reference.wav"
cache = "caches/my_voice.pt"
text = "你好，这是用参考音频音色生成的中文语音。"
output = "outputs/result.wav"
language = "Chinese"
device = "cuda:0"
dtype = "bfloat16"
attn_implementation = "eager"
```

For mixed Chinese, English, and Japanese text, use:

```toml
language = "Auto"
```

## Usage

Run with the config file:

```bash
.venv/bin/python scripts/voice_clone_cached.py \
  --config configs/voice_clone_cached.toml
```

Override config values from the command line:

```bash
.venv/bin/python scripts/voice_clone_cached.py \
  --config configs/voice_clone_cached.toml \
  --text "This line overrides the configured text." \
  --output outputs/override.wav
```

Run with explicit arguments:

```bash
.venv/bin/python scripts/voice_clone_cached.py \
  --model models/Qwen3-TTS-12Hz-1.7B-Base \
  --ref-audio samples/reference.wav \
  --cache caches/my_voice.pt \
  --text "你好，这是测试文本。" \
  --output outputs/result.wav \
  --language Chinese \
  --device cuda:0
```

Force cache rebuild:

```bash
.venv/bin/python scripts/voice_clone_cached.py \
  --config configs/voice_clone_cached.toml \
  --force-rebuild
```

## Fine-Tune a Custom Voice

Qwen3-TTS fine-tuning expects short audio clips with exact transcripts. Each line should contain an audio path, transcript text, and a reference audio path:

```json
{"audio":"data/M3/3星结束行动.wav","text":"少し待て。組織サンプルを採取する必要がある。","ref_audio":"data/M3/交谈1.wav"}
```

The fine-tuning dataset loader requires 24 kHz mono WAV files. Keep the original dataset unchanged and create a 24 kHz copy:

```bash
.venv/bin/python scripts/resample_dataset_24k.py \
  --input_jsonl data/M3/train_raw.jsonl \
  --output_dir data/M3_24k \
  --output_jsonl train_raw.jsonl \
  --sample_rate 24000
```

Generate 12 Hz audio codes for training:

```bash
.venv/bin/python finetuning/prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path models/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl data/M3_24k/train_raw.jsonl \
  --output_jsonl data/M3_24k/train_with_codes.jsonl
```

Run a low-memory 1.7B fine-tune. This mode freezes most model weights and updates the text/codec embeddings and code predictor heads, which is practical on a 16 GB GPU:

```bash
.venv/bin/python finetuning/sft_12hz.py \
  --init_model_path models/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path outputs/finetune_M3_24k_lowmem \
  --train_jsonl data/M3_24k/train_with_codes.jsonl \
  --batch_size 1 \
  --lr 2e-5 \
  --num_epochs 3 \
  --speaker_name M3 \
  --attn_implementation eager \
  --trainable_scope embeddings_and_heads \
  --gradient_accumulation_steps 4 \
  --log_steps 1
```

The final checkpoint from this example is:

```text
outputs/finetune_M3_24k_lowmem/checkpoint-epoch-2
```

## Generate Audio From a Fine-Tuned Voice

Use the CustomVoice generation script with the fine-tuned checkpoint. Speaker names are stored in lowercase, so use `m3` for a checkpoint trained with `--speaker_name M3`:

```bash
.venv/bin/python scripts/generate_custom_voice.py \
  --model outputs/finetune_M3_24k_lowmem/checkpoint-epoch-2 \
  --speaker m3 \
  --language Chinese \
  --text "你好，这是训练后的声音测试。" \
  --output outputs/test_M3.wav \
  --device cuda:0 \
  --attn_implementation eager
```

For mixed Chinese, English, and Japanese text, set:

```bash
--language Auto
```

## Verify Reference Audio

To confirm the full reference audio is loaded before creating a cache:

```bash
.venv/bin/python scripts/voice_clone_cached.py \
  --config configs/voice_clone_cached.toml \
  --inspect-ref-audio
```

The script prints metadata such as:

```text
Reference audio #0: sr=24000, samples=11520000, duration=480.000s (8.00min), estimated_12hz_frames=6000.0
```

When a cache is created, the same metadata is stored as `ref_audio_metadata` in the cache payload.

## Smoke Test

The smoke test validates cache creation and reuse without loading Qwen3-TTS:

```bash
UV_CACHE_DIR=.uv-cache uv run --no-sync python scripts/smoke_test_voice_clone_cached.py
```

## Notes

- The no-transcript mode is convenient but may be less accurate than Qwen3-TTS ICL mode with reference text.
- Use clean, single-speaker reference audio. A short clean clip is usually better than a long noisy recording.
- For reference audio, 3 to 15 seconds is usually enough. Long audio can work but increases first-run cache creation time.
- If running through Codex, GPU commands may require sandbox-external approval. In a normal terminal, `--device cuda:0` is enough when `torch.cuda.is_available()` is true.

## Compliance

Only clone or synthesize voices that you own or have permission to use. Do not use this project to impersonate people, evade consent, or generate unlawful, deceptive, or harmful audio.

## License

This wrapper follows the upstream Qwen3-TTS repository license. See [LICENSE](LICENSE).
