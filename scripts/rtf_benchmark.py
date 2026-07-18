#!/usr/bin/env python3
"""CosyVoice RTF Benchmark — Docker vLLM 0.18.0 + CosyVoice2 (pretrained_models).

Usage:
  docker run --rm --name cosy-rtf \
    --device /dev/davinci5 --device /dev/davinci_manager \
    --device /dev/devmm_svm --device /dev/hisi_hdc \
    --shm-size=8g --network host \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /path/to/CosyVoice:/workspace/CosyVoice \
    -v /path/to/pretrained_models:/models/CosyVoice3 \
    -v ./scripts/rtf_benchmark.py:/workspace/rtf_benchmark.py \
    cosyvoice-rtf:vllm-0.18.0-cosyvoice2 \
    python3 -u /workspace/rtf_benchmark.py

Expected: Overall RTF ~0.68 (1.5x realtime) on Ascend 910B4.
"""
import os
import sys
import time
import json
import torch
import torch_npu
import soundfile as sf
import torchaudio

torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False


def patched_load_wav(w, **kw):
    data, sr = sf.read(w, dtype="float32")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    else:
        data = data.T
    return torch.from_numpy(data.copy()), sr


torchaudio.load = patched_load_wav

sys.path.insert(0, "/workspace/CosyVoice")
sys.path.insert(0, "/workspace/CosyVoice/third_party/Matcha-TTS")
from cosyvoice.cli.cosyvoice import AutoModel

MODEL_DIR = "/models/CosyVoice3"
REF_WAV = "/workspace/CosyVoice/asset/zero_shot_prompt.wav"
REF_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"

TEST_TEXTS = [
    ("你好。", "ultra_short"),
    ("你好，今天天气真不错。", "short"),
    ("欢迎使用实时语音合成系统。", "short"),
    ("八百标兵奔北坡，北坡炮兵并排跑。炮兵怕把标兵碰，标兵怕碰炮兵炮。", "medium"),
    ("今天天气真好，适合出去散步。", "medium"),
    ("床前明月光，疑是地上霜。举头望明月，低头思故乡。", "long"),
    ("数字人技术融合了语音合成、面部驱动、自然语言处理等多个AI模块。", "long"),
]


def main():
    print(f"Loading {MODEL_DIR} vLLM=True...", flush=True)
    t0 = time.perf_counter()
    model = AutoModel(model_dir=MODEL_DIR, load_trt=False, load_vllm=True, fp16=False)
    load_t = time.perf_counter() - t0
    print(f"Loaded in {load_t:.1f}s sr={model.sample_rate} device={model.model.device}", flush=True)

    print("Warmup x3...", flush=True)
    for i in range(3):
        for _ in model.inference_zero_shot("你好。", REF_TEXT, REF_WAV, stream=False):
            pass
        torch.npu.synchronize()
    print("Warmup done.", flush=True)

    results = []
    for text, cat in TEST_TEXTS:
        t0 = time.perf_counter()
        for out in model.inference_zero_shot(text, REF_TEXT, REF_WAV, stream=False):
            audio = out["tts_speech"]
        torch.npu.synchronize()
        dt = time.perf_counter() - t0
        dur = audio.shape[1] / model.sample_rate
        rtf = dt / dur if dur > 0 else 0
        s = "✅" if rtf < 1.0 else "⚠️"
        print(f"  [{cat:>12s}] dur={dur:5.2f}s proc={dt:5.2f}s RTF={rtf:.3f} {s}  | {text[:40]}",
              flush=True)
        results.append({
            "text": text, "cat": cat, "audio_s": round(dur, 3),
            "proc_s": round(dt, 3), "rtf": round(rtf, 3), "realtime": rtf < 1.0,
        })

    ta = sum(r["audio_s"] for r in results)
    tp = sum(r["proc_s"] for r in results)
    ort = tp / ta if ta > 0 else 0
    rc = sum(1 for r in results if r["realtime"])

    print(f"\n{'='*60}")
    print(f"  OVERALL RTF BENCHMARK — vLLM 0.18.0 + CosyVoice2")
    print(f"  Load: {load_t:.1f}s  Total audio: {ta:.2f}s  Total proc: {tp:.2f}s")
    print(f"  Overall RTF: {ort:.3f}  Speed: {ta/tp:.2f}x realtime")
    print(f"  Mean RTF: {sum(r['rtf'] for r in results)/len(results):.3f}")
    print(f"  Realtime: {rc}/{len(results)}")
    print(f"  {'✅ REALTIME' if ort < 1.0 else '⚠️ NOT REALTIME'}")

    with open("/workspace/rtf_result.json", "w") as f:
        json.dump({
            "model": MODEL_DIR, "overall_rtf": round(ort, 3), "load_s": round(load_t, 1),
            "total_audio_s": round(ta, 3), "total_proc_s": round(tp, 3),
            "speed_vs_realtime": round(ta / tp, 2) if tp else 0, "results": results,
        }, f, ensure_ascii=False, indent=2)
    print("\nSaved: /workspace/rtf_result.json")


if __name__ == "__main__":
    main()
