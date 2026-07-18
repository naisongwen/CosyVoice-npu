# CosyVoice3 Ascend NPU Docker TTS 部署技术文档

## 概述

本文档记录了将 CosyVoice3 语音合成模型部署到 Ascend 910B4 NPU 的 Docker 容器中，
并达到 **RTF 0.68（1.5 倍实时）** 的完整过程。

| 项目 | 值 |
|------|-----|
| 模型 | Fun-CosyVoice3-0.5B (CosyVoice2 格式) |
| 基础镜像 | `quay.io/ascend/vllm-ascend:v0.18.0` |
| 最终镜像 | `cosyvoice-rtf:vllm-0.18.0-cosyvoice2` (16.6GB) |
| NPU | Ascend 910B4-1 (64GB HBM) |
| 推理引擎 | vLLM 0.18.0 + vllm-ascend |
| 服务框架 | FastAPI + uvicorn |
| API 协议 | OpenAI 兼容 `/v1/audio/speech` |

---

## 一、镜像构建

### 1.1 基础镜像选择

选择 `quay.io/ascend/vllm-ascend:v0.18.0`，原因：

- 内核已预编译 vLLM + vllm-ascend（NPU 后端插件）
- 内置 torch 2.9.0 + torch_npu 2.9.0
- CANN 8.5.1 环境已配置
- 支持 PIECEWISE ACL Graph 编译模式

### 1.2 依赖安装

```dockerfile
FROM quay.io/ascend/vllm-ascend:v0.18.0

RUN pip install --no-cache-dir \
    hyperpyyaml soundfile wget wetext pyworld \
    inflect hydra-core ruamel.yaml pyarrow \
    omegaconf gdown onnxruntime \
    x-transformers matplotlib openai-whisper conformer \
    diffusers lightning
```

关键依赖说明：

| 包 | 用途 |
|---|------|
| `hyperpyyaml` | CosyVoice 配置解析 |
| `pyworld` | 基频提取 (F0) |
| `wetext` | 中文文本前端处理 |
| `onnxruntime` | 语音 tokenizer 推理 |
| `x-transformers` | DiT flow 的 RotaryEmbedding |
| `openai-whisper` | 语音特征提取 |
| `conformer` | Matcha-TTS 解码器组件 |
| `diffusers` | flow_matching 模块 |

### 1.3 vLLM 模型注册插件

vLLM 通过 entry_points 机制发现自定义模型。需要创建插件文件：

**`cosyvoice_vllm_plugin.py`**：
```python
def register():
    import sys, os
    root = os.environ.get("COSYVOICE_ROOT", "/workspace/CosyVoice")
    if root not in sys.path:
        sys.path.insert(0, root)
    matcha = os.path.join(root, "third_party", "Matcha-TTS")
    if os.path.isdir(matcha) and matcha not in sys.path:
        sys.path.insert(0, matcha)
    from vllm import ModelRegistry
    from cosyvoice.vllm.cosyvoice2 import CosyVoice2ForCausalLM
    ModelRegistry.register_model("CosyVoice2ForCausalLM", CosyVoice2ForCausalLM)
```

**entry_points 元数据** (`site-packages/cosyvoice_vllm_plugin-0.1.0.dist-info/entry_points.txt`)：
```ini
[vllm.general_plugins]
cosyvoice = cosyvoice_vllm_plugin:register
```

> **重要**：vLLM 的 EngineCore 在独立子进程中运行，通过 `importlib.metadata.entry_points(group="vllm.general_plugins")`
> 自动发现插件。如果仅通过 `ModelRegistry.register_model()` 在主进程中注册，
> 子进程将无法识别 `CosyVoice2ForCausalLM`，导致 `ValueError: Model architectures ['CosyVoice2ForCausalLM'] are not supported`。

### 1.4 最终镜像

```bash
docker images cosyvoice-rtf:vllm-0.18.0-cosyvoice2
# REPOSITORY     TAG                              SIZE
# cosyvoice-rtf  vllm-0.18.0-cosyvoice2           16.6GB
```

---

## 二、代码补丁

为了在 Ascend NPU 上正常运行，需要对 CosyVoice 代码做以下修改：

### 2.1 generator.py — NPU _istft 回退

**文件**: `cosyvoice/hifigan/generator.py`

**问题**：`torch.istft` 在 NPU 上不支持 `complex` 类型的 window tensor。

**修复**：检测 NPU 设备，将 complex tensor 和 window 回退到 CPU 计算：

```python
def _istft(self, magnitude, phase):
    magnitude = torch.clip(magnitude, max=1e2)
    device = magnitude.device
    is_npu = device.type == "npu"
    real = magnitude * torch.cos(phase)
    img = magnitude * torch.sin(phase)
    if is_npu:
        complex_tensor = torch.complex(real.cpu(), img.cpu())
        window_xpu = self.stft_window.cpu()
    else:
        complex_tensor = torch.complex(real, img)
        window_xpu = self.stft_window.to(device)
    inverse_transform = torch.istft(
        complex_tensor, self.istft_params["n_fft"],
        self.istft_params["hop_len"], self.istft_params["n_fft"],
        window=window_xpu
    )
    if is_npu:
        inverse_transform = inverse_transform.to(device)
    return inverse_transform
```

### 2.2 model.py — GPU 内存利用率

**文件**: `cosyvoice/cli/model.py`

```python
# 修改前
gpu_memory_utilization=0.2

# 修改后
gpu_memory_utilization=0.5, enable_prefix_caching=True
```

### 2.3 cosyvoice.py — typing 兼容

**文件**: `cosyvoice/cli/cosyvoice.py`

```python
# 修改前
from typing import Optional

# 修改后
from typing import Optional, Union
```

### 2.4 pyworld — pkg_resources 兼容

**问题**：Python 3.11+ 中 `setuptools>=70` 移除了 `pkg_resources`。

**修复**：将 `pyworld/__init__.py` 中的 `import pkg_resources` 替换为
`from importlib.metadata import version`。

---

## 三、启动命令

```bash
docker run -d --name cosyvoice-tts \
  --device /dev/davinci5 \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  --shm-size=8g \
  --network host \
  --restart=unless-stopped \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v /data/avatar-stream/CosyVoice:/workspace/CosyVoice \
  -v /data/avatar-stream/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B:/models/CosyVoice3 \
  -p 58099:58099 \
  cosyvoice-rtf:vllm-0.18.0-cosyvoice2 \
  python3 -u /workspace/CosyVoice/docker/cosyvoice3_server.py \
    --port 58099 --model_dir /models/CosyVoice3 --load_vllm
```

### 参数说明

| 参数 | 说明 |
|------|------|
| `--device /dev/davinci5` | 分配 NPU 5 |
| `--device /dev/davinci_manager` | NPU 管理设备 |
| `--device /dev/devmm_svm` | NPU 虚拟内存 |
| `--device /dev/hisi_hdc` | NPU 驱动通信 |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | **必须**，NPU 不支持 fork |
| `--shm-size=8g` | 共享内存（多进程通信需要） |
| `--load_vllm` | 启用 vLLM 推理引擎 |

---

## 四、API 接口

### 4.1 健康检查

```bash
GET /health

# Response
{
    "status": "ok",
    "model_loaded": true,
    "npu": true,
    "vllm": true
}
```

### 4.2 语音合成

OpenAI TTS 兼容的 `/v1/audio/speech` 端点：

```bash
POST /v1/audio/speech
Content-Type: application/json

{
    "input": "你好，今天天气真不错。",
    "ref_audio": "/workspace/CosyVoice/asset/zero_shot_prompt.wav",
    "ref_text": "希望你以后能够做的比我还好呦。",
    "response_format": "wav"     # "wav" 完整文件 / "pcm" 流式 int16
}
```

#### 流式调用 (Python)

```python
import requests

def tts_stream(text, ref_audio, ref_text, server="http://127.0.0.1:58099"):
    r = requests.post(f"{server}/v1/audio/speech", json={
        "input": text,
        "ref_audio": ref_audio,
        "ref_text": ref_text,
        "response_format": "pcm",
    }, stream=True, timeout=120)

    for chunk in r.iter_content(chunk_size=1920):
        if chunk:
            yield chunk  # int16 PCM, 24kHz, mono
```

#### 非流式调用 (cURL)

```bash
curl -X POST http://localhost:58099/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "你好世界",
    "ref_audio": "/workspace/CosyVoice/asset/zero_shot_prompt.wav",
    "ref_text": "希望你以后能够做的比我还好呦。",
    "response_format": "wav"
  }' -o output.wav
```

---

## 五、性能指标

### 5.1 RTF 基准测试

| 文本长度 | 音频时长 | 处理耗时 | RTF | 实时? |
|---------|---------|---------|-----|:---:|
| 你好 (2字) | 0.68s | 2.30s | 3.39 | ❌ |
| 今天天气不错 (10字) | 2.88s | 3.21s | 1.12 | ❌ |
| 欢迎使用实时合成 (12字) | 2.84s | 3.16s | 1.11 | ❌ |
| 八百标兵... (32字) | 11.92s | 6.96s | **0.58** | ✅ |
| 床前明月光... (20字) | 11.12s | 6.61s | **0.59** | ✅ |
| 数字人技术融合... (76字) | 23.20s | 12.56s | **0.54** | ✅ |

| 总体 | 值 |
|------|-----|
| **Overall RTF** | **0.68** |
| **合成速度** | **1.5 倍实时** |
| 模型加载时间 | ~120s (首次，含 ACL graph 编译) |
| 后续加载 | ~60s (graph 缓存命中) |

### 5.2 显存占用

| 组件 | 大小 | 说明 |
|------|------|------|
| LLM 权重 | 0.7 GB | 0.5B 参数 bfloat16 |
| KV Cache | 29.7 GB | 预分配 2,592,768 tokens |
| Flow 模型 | 2.8 GB | DiT flow + encoder |
| ACL Graph | ~4 GB | torch.compile 缓存 |
| **总计** | **~38 GB** | 64GB HBM 的 59% |

### 5.3 与旧方案对比

| 指标 | 旧方案 (NPU direct) | 新方案 (Docker vLLM) | 提升 |
|------|:---:|:---:|:---:|
| RTF | ~1.7 | **0.68** | **2.5x** |
| 10字文本延迟 | ~9s | ~3s | **3x** |
| 显存 | 7.3 GB | 38 GB | - |
| 部署方式 | host conda env | Docker 容器 | 可移植 |

---

## 六、踩坑记录

### 坑 1：用错了模型格式

**现象**：RTF 稳定在 2.4-2.8，始终无法达到实时。

**原因**：`weights/Fun-CosyVoice3-0.5B-2512` 是 CosyVoice3 格式
（`cosyvoice3.yaml` → `CosyVoice3LM` + `CausalMaskedDiffWithDiT` + `CausalHiFTGenerator`），
DiT flow 推理极慢。

**解决**：使用 `pretrained_models/Fun-CosyVoice3-0.5B`
（`cosyvoice2.yaml` → `CosyVoice2LM` + `CausalMaskedDiffWithXvec` + `HiFTGenerator`），
Xvec flow 推理速度快数倍。

### 坑 2：host 代码缺少 NPU 补丁

**现象**：Docker 容器挂载了 host 的 CosyVoice 代码，容器内原始的补丁被覆盖。

**原因**：`generator.py` 的 `_istft` 方法使用了 `torch.complex(real, img)` 和
`window.to(device)`，而 Ascend NPU 不支持 complex tensor 上的某些操作。

**解决**：在 host 的 `generator.py` 中手动应用 NPU 回退补丁（见 2.1 节），
并提交到 Git。

### 坑 3：vLLM 版本兼容性矩阵

| vLLM 版本 | NPU 支持 | CosyVoice 兼容 | 备注 |
|:---:|:---:|:---:|------|
| 0.11.0 (标准版) | ❌ 无 NPU 平台 | ✅ | `Device string must not be empty` |
| 0.13.0 (vllm-ascend) | ✅ | ✅ | 旧版 ACL graph，加载较慢 |
| 0.17.0rc1 (vllm-ascend) | ✅ | ❌ | `get_device_capability` 返回 None |
| **0.18.0 (vllm-ascend)** | ✅ | ✅ | **推荐**，PIECEWISE 编译 |

### 坑 4：EngineCore 子进程找不到模型

**现象**：
```
ValueError: Model architectures ['CosyVoice2ForCausalLM'] are not supported for now.
```

**原因**：vLLM EngineCore 在独立子进程中运行，仅在主进程通过
`ModelRegistry.register_model()` 注册模型对子进程不可见。

**解决**：必须通过 `entry_points` 机制注册 — 创建
`cosyvoice_vllm_plugin-0.1.0.dist-info/entry_points.txt` 并放入 site-packages，
使 vLLM 自动在所有进程中发现并加载插件。

### 坑 5：NPU 设备映射遗漏

**现象**：`RuntimeError: Invalid device ID` / `aclInit error code 107001`

**原因**：仅 `--device /dev/davinci5` 不够，缺少管理设备和虚拟内存设备。

**正确映射**：
```bash
--device /dev/davinci5        # 计算卡
--device /dev/davinci_manager  # 管理设备
--device /dev/devmm_svm        # NPU 虚拟内存
--device /dev/hisi_hdc         # 驱动通信
```

### 坑 6：VLLM_WORKER_MULTIPROC_METHOD

**现象**：`RuntimeError: Cannot re-initialize NPU in forked subprocess`

**原因**：NPU 驱动不支持 fork，默认的 multiprocessing start method 是 fork。

**解决**：设置环境变量 `VLLM_WORKER_MULTIPROC_METHOD=spawn`。

### 坑 7：Python 3.12 不支持 vllm-ascend

**现象**：尝试在 host 的 Python 3.12 conda 环境中安装 vllm-ascend 失败。

**原因**：`vllm-ascend` 需要编译 C 扩展，且 `setup.py` 中明确限制
`python_requires=">=3.10"`，且部分依赖（如 torch_npu 特定版本）不支持 3.12。

**解决**：使用 Docker（Python 3.11.14），或在 host 上创建 Python 3.11 conda 环境
（但 host 上 torch_npu .so 与 vllm-ascend .so 的 CANN 路径不一致导致 libop_plugin_atb.so 加载失败）。

### 坑 8：pip 依赖地狱

**现象**：`torchaudio` 被错误升级为 CUDA 版本 → `libcudart.so.13` 找不到。
`setuptools` 升级到 83 → `pkg_resources` 被移除 → `pyworld` 崩溃。
`torch` 被 transitive dep 升级到 2.13.0 CUDA → 段错误。

**解决**：
1. 使用 Pytorch CPU wheel 源安装 `torch`、`torch_npu`、`torchaudio`
2. 用 constraints.txt 锁定关键包版本：
   ```
   setuptools==69.5.1
   torch==2.9.0
   torch_npu==2.9.0
   numpy==1.26.4
   ```

### 坑 9：完整设备映射才能用 NPU

首次测试时只传了 `--device /dev/davinci5` 和 `/dev/davinci_manager`，
导致 NPU 初始化超时。对照正在运行的 Qwen 容器后发现还缺
`/dev/devmm_svm`（NPU 虚拟内存设备）和 `/dev/hisi_hdc`（驱动通信设备）。

---

## 七、项目文件结构

```
CosyVoice/
├── docker/
│   ├── Dockerfile.ascend              # 镜像构建文件
│   ├── ascend_entrypoint.sh           # CANN 环境初始化
│   ├── cosyvoice3_server.py           # FastAPI TTS 服务
│   ├── cosyvoice_vllm_plugin.py       # vLLM 模型注册插件
│   └── README.md                      # 部署文档
├── tts/backends/
│   └── cosy_voice_backend.py          # 客户端后端（内嵌/服务端双模）
├── scripts/
│   └── rtf_benchmark.py               # RTF 基准测试脚本
├── cosyvoice/
│   ├── cli/
│   │   ├── cosyvoice.py               # [已修改] Union typing 兼容
│   │   └── model.py                   # [已修改] gpu_memory_utilization
│   ├── hifigan/
│   │   └── generator.py               # [已修改] NPU _istft 回退
│   └── ...
└── docs/
    └── DOCKER_TTS_DEPLOY.md            # 本文档
```

---

## 八、运维命令

```bash
# 查看服务状态
curl http://localhost:58099/health

# 查看日志
docker logs -f cosyvoice-tts

# 查看 NPU 显存
npu-smi info | grep -A6 "^| 5"

# 重启服务
docker restart cosyvoice-tts

# 更新镜像后重新部署
docker rm -f cosyvoice-tts
docker run -d --name cosyvoice-tts ... cosyvoice-rtf:vllm-0.18.0-cosyvoice2

# RTF 基准测试
docker run --rm --name cosy-rtf \
  --device /dev/davinci5 --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  --shm-size=8g -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
  -v $(pwd):/workspace/CosyVoice \
  -v $(pwd)/pretrained_models/Fun-CosyVoice3-0.5B:/models/CosyVoice3 \
  -v $(pwd)/scripts/rtf_benchmark.py:/workspace/rtf_benchmark.py \
  cosyvoice-rtf:vllm-0.18.0-cosyvoice2 \
  python3 -u /workspace/rtf_benchmark.py
```

---

## 九、Git 提交历史

| Commit | 说明 |
|--------|------|
| `5f0bf40` | NPU 性能补丁 (generator, model, cosyvoice) |
| `d4c5935` | Docker 部署文件 (Dockerfile, server, plugin) |
| `c93138e` | OpenAI `/v1/audio/speech` 标准化 + backend 精简 |

仓库地址：`git@github.com:naisongwen/CosyVoice-npu.git`
