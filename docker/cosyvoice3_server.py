#!/usr/bin/env python3
"""
CosyVoice3 Ascend NPU TTS Server (FastAPI + vLLM)

OpenAI-compatible endpoint:
    POST /v1/audio/speech
    - Streaming: response_format="pcm" → chunked raw int16 PCM
    - Non-streaming: response_format="wav" → single WAV file

Supports two inference modes:
    --load_vllm   LLM on NPU via vllm-ascend, flow+hift on NPU/CPU  (RTF ~0.68)
    (default)     llm+flow on NPU direct, hift on CPU
"""
import argparse
import asyncio
import io
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, StreamingResponse

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("cosyvoice3-server")

# ─── NPU init ──────────────────────────────────────────────────
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

# Patch torchaudio.load → soundfile (avoids SoX issues on NPU)
import torchaudio


def _patched_load_wav(w, **kw):
    data, sample_rate = sf.read(w, dtype="float32")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    else:
        data = data.T
    return torch.from_numpy(data.copy()), sample_rate


torchaudio.load = _patched_load_wav

from cosyvoice.cli.cosyvoice import AutoModel
from cosyvoice.cli.model import CosyVoice3Model

# ─── Streaming helpers ─────────────────────────────────────────
SAMPLE_RATE = 24000
STREAM_CHUNK_SAMPLES = 960  # 40ms at 24kHz
STREAM_MEDIA_TYPE = "audio/pcm;rate=24000;channels=1"

# ─── Server ────────────────────────────────────────────────────
pipeline: Optional[AutoModel] = None
_use_vllm = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, _use_vllm
    args = app.state.args
    _use_vllm = getattr(args, "load_vllm", False)

    logger.info("Loading CosyVoice3 from %s (vLLM=%s)", args.model_dir, _use_vllm)
    t0 = time.time()
    pipeline = AutoModel(
        model_dir=args.model_dir, load_trt=False, load_vllm=_use_vllm, fp16=False
    )
    logger.info("Model loaded in %.1fs", time.time() - t0)

    if not _use_vllm and _HAS_NPU:
        _migrate_to_npu()

    logger.info("Server ready on port %d", args.port)
    yield
    logger.info("Shutting down")


app = FastAPI(title="CosyVoice3 Ascend NPU TTS", lifespan=lifespan)


# ─── NPU direct migration ──────────────────────────────────────
def _migrate_to_npu():
    NPU = torch.device("npu:0")
    CPU = torch.device("cpu")
    pipeline.model.llm.to(NPU, dtype=torch.float16)
    pipeline.model.flow.to(NPU, dtype=torch.float16)
    pipeline.model.hift.to(CPU).float()
    pipeline.model.device = NPU

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
    logger.info("NPU token2wav bridge patched")


# ─── OpenAI-compatible API ─────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": pipeline is not None, "npu": _HAS_NPU, "vllm": _use_vllm}


@app.post("/v1/audio/speech")
async def create_speech(payload: dict):
    """OpenAI TTS-compatible endpoint.  Supports both streaming PCM and WAV."""
    if pipeline is None:
        raise HTTPException(503, "not ready")

    # ── parse input ──
    text = str(payload.get("input", payload.get("text", ""))).strip()
    if not text:
        raise HTTPException(400, "text required")

    ref_text = str(payload.get("ref_text", "")).strip()
    ref_audio = str(payload.get("ref_audio", ""))
    ref_b64 = payload.get("ref_audio_b64", "")
    response_format = str(payload.get("response_format", "wav")).lower()

    if not ref_text:
        raise HTTPException(400, "ref_text required")
    if not ref_audio and not ref_b64:
        raise HTTPException(400, "ref_audio required")

    # decode base64 ref_audio if needed
    if ref_b64 and not ref_audio:
        import base64 as b64
        import tempfile
        raw = b64.b64decode(ref_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(raw)
            ref_audio = f.name

    if not os.path.exists(ref_audio):
        raise HTTPException(400, f"ref_audio not found: {ref_audio}")

    # ensure <|endofprompt|> token
    if "<|endofprompt|>" not in ref_text:
        ref_text = "You are a helpful assistant.<|endofprompt|>" + ref_text

    # ── generate ──
    loop = asyncio.get_event_loop()
    t0 = time.perf_counter()

    if response_format == "pcm":
        return StreamingResponse(
            _stream_pcm(text, ref_text, ref_audio, t0, loop),
            media_type=STREAM_MEDIA_TYPE,
            headers={"X-Sample-Rate": str(SAMPLE_RATE)},
        )

    # default: WAV
    try:
        audio = await loop.run_in_executor(None, _generate, text, ref_text, ref_audio)
    except Exception:
        logger.exception("Generation failed")
        raise HTTPException(500, "generation error")

    dt = time.perf_counter() - t0
    dur = len(audio) / SAMPLE_RATE
    logger.info("TTS %.2fs → %.2fs  RTF=%.3f  text=%s", dur, dt, dt / dur, text[:40])

    buf = io.BytesIO()
    sf.write(buf, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return Response(buf.read(), media_type="audio/wav")


async def _stream_pcm(text, ref_text, ref_audio, t0, loop):
    """Async generator yielding int16 PCM byte chunks."""
    audio = await loop.run_in_executor(None, _generate, text, ref_text, ref_audio)
    dt = time.perf_counter() - t0
    dur = len(audio) / SAMPLE_RATE
    logger.info("TTS %.2fs → %.2fs  RTF=%.3f  text=%s", dur, dt, dt / dur, text[:40])

    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    buf = pcm.tobytes()
    for i in range(0, len(buf), STREAM_CHUNK_SAMPLES * 2):
        yield buf[i : i + STREAM_CHUNK_SAMPLES * 2]
        await asyncio.sleep(0)  # yield to event loop


def _generate(text, ref_text, ref_audio_path):
    with torch.inference_mode():
        for out in pipeline.inference_zero_shot(
            text, ref_text, ref_audio_path, stream=False
        ):
            audio = out["tts_speech"]
    return audio.cpu().numpy().flatten()


# ─── entrypoint ────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=58099)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--load_vllm", action="store_true")
    args = parser.parse_args()
    app.state.args = args
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
