#!/usr/bin/env python3
"""
CosyVoice3 Ascend NPU TTS Server (FastAPI + vLLM)
- vLLM mode (--load_vllm): LLM runs on NPU via vllm-ascend, flow+hift on NPU/CPU
- Direct mode (default): llm+flow on NPU, hift on CPU

RTF ~0.68 (1.5x realtime) on Ascend 910B4 with vLLM mode.
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
from fastapi.responses import Response

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
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

# Patch torchaudio.load to use soundfile (avoids SoX issues on NPU)
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

# ─── Server ────────────────────────────────────────────────────
pipeline: Optional[AutoModel] = None
_load_vllm = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline, _load_vllm

    args = app.state.args
    _load_vllm = getattr(args, "load_vllm", False)

    logger.info("Loading CosyVoice3 from %s (vLLM=%s)...", args.model_dir, _load_vllm)
    t0 = time.time()
    pipeline = AutoModel(model_dir=args.model_dir, load_trt=False, load_vllm=_load_vllm, fp16=False)
    logger.info("Model loaded in %.1fs", time.time() - t0)

    if not _load_vllm:
        # NPU direct migration
        if _HAS_NPU:
            NPU = torch.device("npu:0")
            CPU = torch.device("cpu")
            pipeline.model.llm.to(NPU, dtype=torch.float16)
            pipeline.model.flow.to(NPU, dtype=torch.float16)
            pipeline.model.hift.to(CPU).float()
            pipeline.model.device = NPU
            logger.info("Migrated llm+flow to NPU fp16, hift on CPU")

            # Patch token2wav for NPU
            @torch.inference_mode()
            def _npubridge_token2wav(self, token, prompt_token, prompt_feat, embedding,
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

            CosyVoice3Model.token2wav = _npubridge_token2wav
            logger.info("NPU token2wav bridge patched")

    logger.info("Server ready on port %d", args.port)
    yield
    logger.info("Shutting down")


app = FastAPI(title="CosyVoice3 Ascend NPU TTS", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": pipeline is not None, "npu": _HAS_NPU, "vllm": _load_vllm}


@app.post("/v1/audio/speech")
async def create_speech(payload: dict):
    if pipeline is None:
        raise HTTPException(503, "not ready")

    text = str(payload.get("input", payload.get("text", ""))).strip()
    if not text:
        raise HTTPException(400, "text required")

    ref_audio = str(payload.get("ref_audio", ""))
    ref_b64 = payload.get("ref_audio_b64", "")
    ref_text = str(payload.get("ref_text", ""))

    if not ref_text:
        raise HTTPException(400, "ref_text required")
    if not ref_audio and not ref_b64:
        raise HTTPException(400, "ref_audio required")

    # Decode base64 ref if provided
    if ref_b64 and not ref_audio:
        import base64
        import tempfile

        raw = base64.b64decode(ref_b64)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(raw)
            ref_audio = f.name

    if not os.path.exists(ref_audio):
        raise HTTPException(400, f"ref_audio not found: {ref_audio}")

    format_type = str(payload.get("response_format", "wav")).lower()

    # CosyVoice3 expects <|endofprompt|> in ref_text
    if "<|endofprompt|>" not in ref_text:
        ref_text = "You are a helpful assistant.<|endofprompt|>" + ref_text

    loop = asyncio.get_event_loop()

    try:
        t0 = time.perf_counter()
        audio = await loop.run_in_executor(None, _generate, text, ref_text, ref_audio)
        dt = time.perf_counter() - t0
    except Exception as e:
        logger.exception("Generation failed")
        raise HTTPException(500, str(e))

    # Save WAV
    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    audio_dur = len(audio) / 24000
    rtf = dt / audio_dur if audio_dur > 0 else 0
    logger.info("Generated %.2fs audio in %.2fs, RTF=%.3f", audio_dur, dt, rtf)

    if format_type == "pcm":
        return Response(wav_bytes[44:], media_type="audio/pcm")
    return Response(wav_bytes, media_type="audio/wav")


def _generate(text, ref_text, ref_audio_path):
    with torch.inference_mode():
        for out in pipeline.inference_zero_shot(text, ref_text, ref_audio_path, stream=False):
            audio = out["tts_speech"]
    return audio.cpu().numpy().flatten()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=58099)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--load_vllm", action="store_true")
    args = parser.parse_args()
    app.state.args = args
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
