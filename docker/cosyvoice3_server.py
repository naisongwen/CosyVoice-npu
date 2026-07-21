#!/usr/bin/env python3
"""
CosyVoice3 Ascend NPU TTS Server (FastAPI + vLLM) — Optimized

OpenAI-compatible: POST /v1/audio/speech

Optimizations applied:
  1. onnxruntime-cann       → campplus + speech_tokenizer ONNX run on NPU
  2. ref extraction cache    → embedding/token/feat cached per ref_audio file
  3. ONNX InferenceSession   → monkey-patched to prefer CANNExecutionProvider
"""
import argparse
import asyncio
import hashlib
import io
import logging
import os
import sys
import time
import threading
from contextlib import asynccontextmanager
from functools import wraps
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cosyvoice3-server")

# ═══════════════════════════════════════════════════════════════
# NPU init
# ═══════════════════════════════════════════════════════════════

try:
    import torch_npu
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False
    _HAS_NPU = torch.npu.is_available()
except ImportError:
    _HAS_NPU = False

COSYVOICE_ROOT = os.environ.get("COSYVOICE_ROOT", "/workspace/CosyVoice")
sys.path.insert(0, COSYVOICE_ROOT)
sys.path.insert(0, os.path.join(COSYVOICE_ROOT, "third_party", "Matcha-TTS"))

import torchaudio

def _patched_load_wav(w, **kw):
    data, sample_rate = sf.read(w, dtype="float32")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    else:
        data = data.T
    return torch.from_numpy(data.copy()), sample_rate

torchaudio.load = _patched_load_wav

# ═══════════════════════════════════════════════════════════════
# ONNX CANN patch
# ═══════════════════════════════════════════════════════════════

_HAS_CANN_ONNX = False
if _HAS_NPU:
    try:
        import onnxruntime as ort
        if "CANNExecutionProvider" in ort.get_available_providers():
            _HAS_CANN_ONNX = True
            _orig_isess = ort.InferenceSession

            class _CANNInferenceSession(ort.InferenceSession):
                def __init__(self, model_path, *a, **kw):
                    kw.setdefault("providers", [
                        ("CANNExecutionProvider", {"device_id": 0}),
                        "CPUExecutionProvider",
                    ])
                    super().__init__(model_path, *a, **kw)

            ort.InferenceSession = _CANNInferenceSession  # type: ignore[assignment]
            logger.info("ONNX → CANNExecutionProvider (NPU acceleration)")
        else:
            logger.warning("CANNExecutionProvider not available; ONNX stays on CPU")
    except Exception as e:
        logger.warning("ONNX CANN patch failed: %s", e)

# ═══════════════════════════════════════════════════════════════
# Ref extraction cache — wraps CosyVoiceFrontEnd methods
# ═══════════════════════════════════════════════════════════════

class RefCache:
    """Thread-safe cache for ref-audio extraction results."""

    def __init__(self, max_size: int = 64):
        self._lock = threading.Lock()
        self._data: dict[str, dict] = {}
        self._max = max_size

    @staticmethod
    def _hash(path: str) -> str:
        s = os.stat(path)
        with open(path, "rb") as f:
            return hashlib.md5(
                f"{s.st_mtime}_{s.st_size}_{f.read(4096)}".encode()
            ).hexdigest()

    def get(self, path: str) -> Optional[dict]:
        k = self._hash(path)
        with self._lock:
            if k in self._data:
                return {kk: v.clone() if hasattr(v, "clone") else v
                        for kk, v in self._data[k].items()}
        return None

    def put(self, path: str, data: dict):
        k = self._hash(path)
        with self._lock:
            if len(self._data) >= self._max:
                oldest = next(iter(self._data))
                del self._data[oldest]
            # Store CPU copies (don't hold NPU memory between requests)
            self._data[k] = {
                kk: v.cpu().clone() if hasattr(v, "clone") and hasattr(v, "cpu") else v
                for kk, v in data.items()
            }

    def stats(self) -> dict:
        with self._lock:
            return {"entries": len(self._data), "max": self._max}

_ref_cache = RefCache()


def _install_cache_patch(frontend):
    """Monkey-patch frontend to cache expensive ref-extraction calls.

    _extract_speech_feat, _extract_speech_token, _extract_spk_embedding
    all read from the same wav file. We cache their results keyed by path.
    """
    _orig_feat = frontend._extract_speech_feat
    _orig_token = frontend._extract_speech_token
    _orig_spk = frontend._extract_spk_embedding

    @wraps(_orig_feat)
    def cached_feat(wav):
        c = _ref_cache.get(wav)
        if c and "speech_feat" in c:
            logger.debug("Cache HIT feat: %s", wav)
            return c["speech_feat"], c["speech_feat_len"]
        return _orig_feat(wav)

    @wraps(_orig_token)
    def cached_token(wav):
        c = _ref_cache.get(wav)
        if c and "speech_token" in c:
            logger.debug("Cache HIT token: %s", wav)
            return c["speech_token"], c["speech_token_len"]
        return _orig_token(wav)

    @wraps(_orig_spk)
    def cached_spk(wav):
        c = _ref_cache.get(wav)
        if c and "embedding" in c:
            logger.debug("Cache HIT embedding: %s", wav)
            return c["embedding"]
        return _orig_spk(wav)

    # Store after extraction
    _orig_frontend_zero_shot = frontend.frontend_zero_shot

    @wraps(_orig_frontend_zero_shot)
    def cached_zero_shot(tts_text, prompt_text, prompt_wav, resample_rate, zero_shot_spk_id):
        cached = _ref_cache.get(prompt_wav)
        if cached is not None and zero_shot_spk_id == "":
            # Use cached ref data, only re-tokenize tts_text
            logger.info("Ref cache HIT: %s", prompt_wav)
            t0 = time.perf_counter()
            tts_token, tts_token_len = frontend._extract_text_token(tts_text)
            prompt_token, prompt_token_len = frontend._extract_text_token(prompt_text)

            if resample_rate == 24000:
                token_len = min(int(cached["speech_feat"].shape[1] / 2),
                                cached["speech_token"].shape[1])
                speech_feat = cached["speech_feat"][:, :2 * token_len]
                speech_feat_len = torch.tensor([2 * token_len], dtype=torch.int32)
                speech_token = cached["speech_token"][:, :token_len]
                speech_token_len = torch.tensor([token_len], dtype=torch.int32)
            else:
                speech_feat = cached["speech_feat"]
                speech_feat_len = cached["speech_feat_len"]
                speech_token = cached["speech_token"]
                speech_token_len = cached["speech_token_len"]

            model_input = {
                "text": tts_token, "text_len": tts_token_len,
                "prompt_text": prompt_token, "prompt_text_len": prompt_token_len,
                "llm_prompt_speech_token": speech_token,
                "llm_prompt_speech_token_len": speech_token_len,
                "flow_prompt_speech_token": speech_token,
                "flow_prompt_speech_token_len": speech_token_len,
                "prompt_speech_feat": speech_feat, "prompt_speech_feat_len": speech_feat_len,
                "llm_embedding": cached["embedding"], "flow_embedding": cached["embedding"],
            }
            dt = time.perf_counter() - t0
            logger.info("Ref cache: saved %.3fs", dt)
            return model_input

        # First call — run original, then cache all ref outputs
        t0 = time.perf_counter()
        result = _orig_frontend_zero_shot(tts_text, prompt_text, prompt_wav,
                                          resample_rate, zero_shot_spk_id)
        dt = time.perf_counter() - t0
        if zero_shot_spk_id == "":
            # Extract ref data again to cache (we already computed them, but
            # let's re-extract explicitly for clean cache storage)
            try:
                sf, sfl = _orig_feat(prompt_wav)
                st, stl = _orig_token(prompt_wav)
                emb = _orig_spk(prompt_wav)
                _ref_cache.put(prompt_wav, {
                    "speech_feat": sf, "speech_feat_len": sfl,
                    "speech_token": st, "speech_token_len": stl,
                    "embedding": emb,
                })
                logger.info("Ref extract: %.3fs  Cache STORE → %s", dt, prompt_wav)
            except Exception as e:
                logger.warning("Failed to cache ref data: %s", e)
        else:
            logger.info("Ref extract: %.3fs (spk_id=%s, not cached)", dt, zero_shot_spk_id)
        return result

    frontend._extract_speech_feat = cached_feat
    frontend._extract_speech_token = cached_token
    frontend._extract_spk_embedding = cached_spk
    frontend.frontend_zero_shot = cached_zero_shot
    logger.info("Ref extraction cache installed (max=%d entries)", _ref_cache._max)


# ═══════════════════════════════════════════════════════════════
# Server
# ═══════════════════════════════════════════════════════════════

pipeline: Optional["AutoModel"] = None
_use_vllm = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, _use_vllm
    args = app.state.args
    _use_vllm = getattr(args, "load_vllm", False)

    from cosyvoice.cli.cosyvoice import AutoModel as _AutoModel

    logger.info("Loading CosyVoice3 from %s (vLLM=%s)", args.model_dir, _use_vllm)
    t0 = time.time()
    pipeline = _AutoModel(model_dir=args.model_dir, load_trt=False, load_vllm=_use_vllm, fp16=False)
    logger.info("Model loaded in %.1fs", time.time() - t0)

    # Install ref extraction cache
    _install_cache_patch(pipeline.frontend)

    if _use_vllm:
        logger.info("vLLM mode active (LLM on NPU)")
    elif _HAS_NPU:
        _migrate_to_npu()

    logger.info("Optimizations: ONNX-CANN=%s  EmbedCache=ON", _HAS_CANN_ONNX)
    logger.info("Server ready on port %d", args.port)
    yield
    logger.info("Shutting down")


app = FastAPI(title="CosyVoice3 Ascend NPU TTS", lifespan=lifespan)

from cosyvoice.cli.cosyvoice import AutoModel  # noqa: E402

SAMPLE_RATE = 24000
STREAM_CHUNK_SAMPLES = 960


def _migrate_to_npu():
    NPU = torch.device("npu:0")
    CPU = torch.device("cpu")
    pipeline.model.llm.to(NPU, dtype=torch.float16)
    pipeline.model.flow.to(NPU, dtype=torch.float16)
    pipeline.model.hift.to(CPU).float()
    pipeline.model.device = NPU

    from cosyvoice.cli.model import CosyVoice3Model

    @torch.inference_mode()
    def _npu_token2wav(self, token, prompt_token, prompt_feat, embedding,
                        token_offset, uuid, stream=False, finalize=False, speed=1.0):
        with torch.npu.amp.autocast(True):
            mel, _ = self.flow.inference(
                token=token.to(self.device, dtype=torch.int32),
                token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                prompt_token=prompt_token.to(self.device),
                prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                prompt_feat=prompt_feat.to(self.device),
                prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                embedding=embedding.to(self.device),
                streaming=stream, finalize=finalize,
            )
        speech, _ = self.hift.inference(speech_feat=mel.cpu().float(), finalize=finalize)
        return speech.cpu()

    CosyVoice3Model.token2wav = _npu_token2wav
    logger.info("NPU token2wav patched")


# ═══════════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════════

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "npu": _HAS_NPU,
        "vllm": _use_vllm,
        "onnx_cann": _HAS_CANN_ONNX,
        "cache": _ref_cache.stats(),
    }


@app.post("/v1/audio/speech")
async def create_speech(payload: dict):
    if pipeline is None:
        raise HTTPException(503, "not ready")

    text = str(payload.get("input", payload.get("text", ""))).strip()
    if not text:
        raise HTTPException(400, "text required")

    ref_text = str(payload.get("ref_text", "")).strip()
    ref_audio = str(payload.get("ref_audio", ""))
    ref_b64 = payload.get("ref_audio_b64", "")
    fmt = str(payload.get("response_format", "wav")).lower()

    if not ref_text:
        raise HTTPException(400, "ref_text required")
    if not ref_b64 and not ref_audio:
        raise HTTPException(400, "ref_audio required")

    if ref_b64:
        import base64 as b64, tempfile
        raw = b64.b64decode(ref_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(raw)
            ref_audio = f.name

    if not os.path.exists(ref_audio):
        raise HTTPException(400, f"ref_audio not found: {ref_audio}")

    # Ensure <|endofprompt|> token
    if "<|endofprompt|>" not in ref_text:
        ref_text = "You are a helpful assistant.<|endofprompt|>" + ref_text

    loop = asyncio.get_event_loop()

    if fmt == "pcm":
        t0 = time.perf_counter()
        audio = await loop.run_in_executor(None, _generate, text, ref_text, ref_audio)
        dt = time.perf_counter() - t0
        dur = len(audio) / SAMPLE_RATE
        logger.info("TTS dur=%.2fs RTF=%.3f text=%s", dur, dt / dur if dur else 0, text[:40])
        return StreamingResponse(
            _pcm_chunks(audio),
            media_type="audio/pcm;rate=24000;channels=1",
            headers={"X-Sample-Rate": str(SAMPLE_RATE)},
        )

    t0 = time.perf_counter()
    audio = await loop.run_in_executor(None, _generate, text, ref_text, ref_audio)
    dt = time.perf_counter() - t0
    dur = len(audio) / SAMPLE_RATE
    logger.info("TTS dur=%.2fs RTF=%.3f text=%s", dur, dt / dur if dur else 0, text[:40])

    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return Response(buf.getvalue(), media_type="audio/wav")


def _generate(text, ref_text, ref_audio_path):
    for out in pipeline.inference_zero_shot(text, ref_text, ref_audio_path, stream=False):
        audio = out["tts_speech"]
    return audio.cpu().numpy().flatten()


async def _pcm_chunks(audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes()
    for i in range(0, len(pcm), STREAM_CHUNK_SAMPLES * 2):
        yield pcm[i : i + STREAM_CHUNK_SAMPLES * 2]
        await asyncio.sleep(0)


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=58099)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--load_vllm", action="store_true")
    parser.add_argument("--cache_size", type=int, default=64)
    args = parser.parse_args()
    _ref_cache._max = args.cache_size
    app.state.args = args
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
