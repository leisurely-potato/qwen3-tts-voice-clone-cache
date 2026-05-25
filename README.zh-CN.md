# Qwen3-TTS 音色缓存工具

[English README](README.md)

本项目基于 QwenLM 开源模型 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) 构建。本仓库只是一个无参考文本音色克隆与缓存复用工作流封装，不包含原始模型权重。

这是一个基于 Qwen3-TTS 1.7B Base 的无参考文本音色克隆工作流。它会从参考音频中提取可复用的 speaker prompt 并保存为缓存，后续生成新文本语音时直接复用缓存，避免重复处理同一段参考音频。

## 项目简介

Qwen3-TTS 支持通过参考音频进行音色克隆。当你只有参考音频、没有对应文本时，本项目使用 `x_vector_only_mode=True`：只提取说话人音色向量，不需要参考音频的转写文本。

第一次运行会创建音色缓存；后续运行会加载缓存并直接生成目标文本音频。

## 功能特性

- 基于 Qwen3-TTS 1.7B Base 的无文本音色克隆
- 支持音色缓存复用
- 支持 TOML 配置文件
- 命令行参数可覆盖配置文件
- 支持 GPU 推理，例如 `cuda:0`
- 支持检查参考音频的时长、采样率、样本数和预估 12.5 Hz 帧数
- 提供不加载模型的本地 smoke test

## 目录结构

```text
configs/voice_clone_cached.toml       配置文件示例
scripts/voice_clone_cached.py         主命令行脚本
scripts/generate_custom_voice.py      使用微调后的 CustomVoice checkpoint 生成音频
scripts/resample_dataset_24k.py       将 JSONL 数据集音频转换为 24 kHz WAV
scripts/smoke_test_voice_clone_cached.py
scripts/README_voice_clone_cached.md  脚本详细说明
```

以下目录是本地产物，不会提交到 Git：

```text
models/
data/
caches/
outputs/
samples/
```

## 环境要求

- Python 3.11+
- 推荐使用 NVIDIA GPU 运行 1.7B 模型
- 已下载 Qwen3-TTS 模型权重，或可以从 ModelScope / Hugging Face 下载
- 使用 `uv` 管理 Python 环境

`flash-attn` 是可选加速项。默认 `eager` attention 也可以运行。安装 `flash-attn` 需要 CUDA Toolkit 开发环境，也就是系统里有 `nvcc` 和 `CUDA_HOME`。

## 安装

创建虚拟环境：

```bash
cd /root/codex/qwen-TTS/Qwen3-TTS
UV_CACHE_DIR=.uv-cache uv venv --python python3.11
```

安装依赖。国内建议使用清华源：

```bash
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python -e . \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

可选安装 FlashAttention：

```bash
UV_CACHE_DIR=.uv-cache uv pip install --python .venv/bin/python flash-attn --no-build-isolation \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

## 下载模型

国内优先使用 ModelScope：

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

也可以使用 Hugging Face 镜像：

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

## 配置

编辑：

```text
configs/voice_clone_cached.toml
```

示例：

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

如果文本中混合中文、英文、日文，建议使用：

```toml
language = "Auto"
```

## 使用方法

使用配置文件运行：

```bash
.venv/bin/python scripts/voice_clone_cached.py \
  --config configs/voice_clone_cached.toml
```

命令行参数可以临时覆盖配置文件：

```bash
.venv/bin/python scripts/voice_clone_cached.py \
  --config configs/voice_clone_cached.toml \
  --text "这次临时换一段文本。" \
  --output outputs/override.wav
```

也可以完全使用命令行参数：

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

强制重新生成缓存：

```bash
.venv/bin/python scripts/voice_clone_cached.py \
  --config configs/voice_clone_cached.toml \
  --force-rebuild
```

## 微调一个自定义音色

Qwen3-TTS 微调需要短音频和逐字准确的文本。每一行应包含音频路径、文本和参考音频路径：

```json
{"audio":"data/M3/3星结束行动.wav","text":"少し待て。組織サンプルを採取する必要がある。","ref_audio":"data/M3/交谈1.wav"}
```

微调数据加载器要求音频是 24 kHz 单声道 WAV。建议保留原始数据不动，另外生成一份 24 kHz 训练副本：

```bash
.venv/bin/python scripts/resample_dataset_24k.py \
  --input_jsonl data/M3/train_raw.jsonl \
  --output_dir data/M3_24k \
  --output_jsonl train_raw.jsonl \
  --sample_rate 24000
```

生成训练需要的 12 Hz audio codes：

```bash
.venv/bin/python finetuning/prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path models/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl data/M3_24k/train_raw.jsonl \
  --output_jsonl data/M3_24k/train_with_codes.jsonl
```

运行 1.7B 低显存微调。这个模式会冻结大部分模型权重，只更新 text/codec embedding 和 code predictor 的 embedding/head/projection，在 16GB 显存的 GPU 上更容易跑通：

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

这个示例最终 checkpoint 是：

```text
outputs/finetune_M3_24k_lowmem/checkpoint-epoch-2
```

## 使用微调后的音色生成音频

使用 CustomVoice 生成脚本即可。注意：checkpoint 中的 speaker 名会以小写保存，所以用 `--speaker_name M3` 训练出来后，生成时使用 `--speaker m3`：

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

如果文本里混合中文、英文、日文，使用：

```bash
--language Auto
```

## 检查参考音频

如果你想确认脚本是否完整读取了参考音频，可以运行：

```bash
.venv/bin/python scripts/voice_clone_cached.py \
  --config configs/voice_clone_cached.toml \
  --inspect-ref-audio
```

日志会输出类似：

```text
Reference audio #0: sr=24000, samples=11520000, duration=480.000s (8.00min), estimated_12hz_frames=6000.0
```

创建缓存时，这些信息也会写入缓存文件的 `ref_audio_metadata` 字段。

## Smoke Test

本地 smoke test 不加载 Qwen3-TTS，只验证缓存创建和复用逻辑：

```bash
UV_CACHE_DIR=.uv-cache uv run --no-sync python scripts/smoke_test_voice_clone_cached.py
```

## 注意事项

- 无参考文本模式更方便，但音色克隆质量通常不如提供参考文本的 ICL 模式。
- 推荐使用清晰、单人、无背景音乐、低噪声的参考音频。
- 参考音频通常 3 到 15 秒已经足够。较长音频可以使用，但第一次创建缓存会更慢。
- 如果通过 Codex 执行 GPU 命令，可能需要授权沙箱外运行。在普通终端中，只要 `torch.cuda.is_available()` 为 true，使用 `--device cuda:0` 即可。

## 合规声明

请只克隆你本人拥有或已获得授权的声音。不要将本项目用于冒充他人、规避授权、欺诈、误导或生成违法有害音频。

## 许可证

本工具遵循上游 Qwen3-TTS 仓库许可证。详见 [LICENSE](LICENSE)。
