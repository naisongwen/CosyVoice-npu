# CosyVoice3 Ascend NPU Docker Deployment

## Requirements

- Ascend 910B series NPU (e.g., 910B4-1)
- CANN 8.5.1+
- Docker with `--device` support

## Quick Start

### 1. Build or pull the image

```bash
docker build -t cosyvoice-rtf:latest -f Dockerfile.ascend .
```

Or use the pre-built image:
```bash
docker pull <registry>/cosyvoice-rtf:vllm-0.18.0-cosyvoice2
```

### 2. Mount model and code

```bash
# Model: pretrained_models/Fun-CosyVoice3-0.5B (CosyVoice2 format)
# Code:  CosyVoice repo with cosyvoice/ module and third_party/Matcha-TTS

MODEL_DIR=/path/to/pretrained_models/Fun-CosyVoice3-0.5B
CODE_DIR=/path/to/CosyVoice
```

### 3. Start the server

```bash
docker run -d --name cosyvoice-tts \
  --device /dev/davinci5 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  --shm-size=8g \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v $CODE_DIR:/workspace/CosyVoice \
  -v $MODEL_DIR:/models/CosyVoice3 \
  -p 58099:58099 \
  cosyvoice-rtf:latest
```

### 4. Test

```bash
curl -X POST http://localhost:58099/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "你好，今天天气真不错。",
    "ref_audio": "/workspace/CosyVoice/asset/zero_shot_prompt.wav",
    "ref_text": "希望你以后能够做的比我还好呦。"
  }' -o output.wav
```

### 5. Run RTF benchmark

```bash
docker run --rm --name cosy-rtf \
  --device /dev/davinci5 --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  --shm-size=8g \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v $CODE_DIR:/workspace/CosyVoice \
  -v $MODEL_DIR:/models/CosyVoice3 \
  -v $CODE_DIR/scripts/rtf_benchmark.py:/workspace/rtf_benchmark.py \
  cosyvoice-rtf:latest \
  python3 -u /workspace/rtf_benchmark.py
```

## Performance

| Metric | Value |
|--------|-------|
| Overall RTF | **0.68** |
| Speed | **1.5x realtime** |
| HBM Usage | ~38 GB (with default KV cache) |
| Model Load Time | ~120s (first run, includes ACL graph compilation) |

## Notes

- The vLLM plugin (`cosyvoice_vllm_plugin.py`) auto-registers `CosyVoice2ForCausalLM` with the vLLM ModelRegistry
- Entry points must be installed in `site-packages/cosyvoice_vllm_plugin-0.1.0.dist-info/` for EngineCore subprocess discovery
- Use `--load_vllm` flag for vLLM mode; omit for NPU direct mode
