# -*- coding: utf-8 -*-
from __future__ import annotations

"""CosyVoice TTS Backend.

Two modes (auto-selected):
  embedded   — load model in-process (used when standalone=false)
  server     — call external /v1/audio/speech endpoint (standalone=true)

The server communicates via OpenAI-compatible POST /v1/audio/speech
with streaming PCM (response_format="pcm").
"""
import asyncio
import base64
import os
import sys
import tempfile
import threading
from types import SimpleNamespace

import numpy as np

from tts import tts_config
from utils.logger import logger

PROJECT_ROOT = tts_config.PROJECT_ROOT
COSYVOICE3_END_PROMPT = "<|endofprompt|>"

# ── embedded singleton ─────────────────────────────────────────
_EMBEDDED_BACKEND = None
_EMBEDDED_BACKEND_LOCK = threading.Lock()

# ── helpers ────────────────────────────────────────────────────

def _patch_torchaudio():
    import torch
    import torchaudio

    def _load(wav, **kw):
        import soundfile as sf
        data, sr = sf.read(wav, dtype="float32")
        if data.ndim == 1:
            data = data.reshape(1, -1)
        else:
            data = data.T
        return torch.from_numpy(data.copy()), sr

    torchaudio.load = _load


def _patch_hift_float64():
    """CosyVoice3 hift uses float64, incompatible with NPU. Force float32."""
    import torch
    from cosyvoice.hifigan.generator import CausalHiFTGenerator

    @torch.inference_mode()
    def _patched(self, speech_feat, finalize=True):
        self.f0_predictor = self.f0_predictor.to(torch.float32)
        f0 = self.f0_predictor(speech_feat.to(torch.float32), finalize=finalize).to(speech_feat)
        s = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        s, _, _ = self.m_source(s)
        s = s.transpose(1, 2)
        if finalize:
            g = self.decode(x=speech_feat, s=s, finalize=finalize)
        else:
            g = self.decode(
                x=speech_feat[:, :, : -self.f0_predictor.condnet[0].causal_padding],
                s=s, finalize=finalize,
            )
        return g, s

    CausalHiFTGenerator.inference = _patched


def _setup_cosyvoice_python_path() -> str | None:
    for repo_dir in [os.path.join(PROJECT_ROOT, "CosyVoice")]:
        if not os.path.isdir(repo_dir):
            continue
        sys.path.insert(0, repo_dir)
        matcha = os.path.join(repo_dir, "third_party", "Matcha-TTS")
        if os.path.isdir(matcha):
            sys.path.insert(0, matcha)
        return repo_dir
    return None


# ── CosyVoiceBackend ───────────────────────────────────────────

class CosyVoiceBackend:
    """Loads and runs CosyVoice model in-process (embedded mode)."""

    def __init__(self, args):
        self.args = args
        self.model = None
        self.sample_rate = 24000
        self._lock = threading.Lock()
        self._repo_dir = None
        self._is_cosyvoice3 = False
        self.server_config = {"venture": "cosy_voice"}

    # ── load ───────────────────────────────────────────────────

    def load(self):
        self._repo_dir = _setup_cosyvoice_python_path()
        _patch_torchaudio()
        _patch_hift_float64()

        from cosyvoice.cli.cosyvoice import AutoModel

        model_dir = getattr(self.args, "model_dir", None) or getattr(self.args, "model", None)
        if not model_dir:
            for c in [
                os.path.join(PROJECT_ROOT, "weights", "Fun-CosyVoice3-0.5B-2512"),
                os.path.join(PROJECT_ROOT, "CosyVoice", "pretrained_models", "Fun-CosyVoice3-0.5B"),
            ]:
                if os.path.exists(os.path.join(c, "cosyvoice3.yaml")) or \
                   os.path.exists(os.path.join(c, "cosyvoice2.yaml")):
                    model_dir = c
                    break
            if not model_dir:
                model_dir = os.path.join(
                    PROJECT_ROOT, "CosyVoice", "pretrained_models", "Fun-CosyVoice3-0.5B"
                )

        logger.info("[CosyVoice] loading model: %s", model_dir)

        load_trt = getattr(self.args, "trt", False)
        load_vllm = getattr(self.args, "vllm", False)
        fp16_val = getattr(self.args, "fp16", False)

        # vllm-ascend not available in host env → fall back to NPU direct
        if load_vllm:
            try:
                from vllm.platforms import current_platform
                if not current_platform.device_type:
                    logger.warning("[CosyVoice] vllm-ascend not detected, falling back to NPU direct")
                    load_vllm = False
            except ImportError:
                logger.warning("[CosyVoice] vllm.platforms missing, falling back to NPU direct")
                load_vllm = False

        self.model = AutoModel(
            model_dir=model_dir, fp16=fp16_val, load_trt=load_trt, load_vllm=load_vllm,
        )
        self._is_cosyvoice3 = hasattr(self.model.model, "token_hop_len")
        self.sample_rate = int(getattr(self.model, "sample_rate", 24000) or 24000)

        logger.info("[CosyVoice] CosyVoice%d sample_rate=%d",
                    3 if self._is_cosyvoice3 else 2, self.sample_rate)
        self.server_config.update({
            "model_dir": model_dir, "sample_rate": self.sample_rate,
            "version": "cosyvoice3" if self._is_cosyvoice3 else "cosyvoice2",
        })
        self._warmup()

    def health(self):
        return {
            "venture": "cosy_voice",
            "model_loaded": self.model is not None,
            "config": self.server_config,
        }

    # ── synthesis ──────────────────────────────────────────────

    async def stream_tts(self, payload: dict):
        """Async stream interface: yields int16 PCM byte chunks."""
        text = str(payload.get("text") or "").strip()
        if not text:
            raise ValueError("text required")

        mode = str(payload.get("mode") or "voice_clone")
        loop = asyncio.get_event_loop()

        if mode == "tts":
            speaker = str(payload.get("speaker") or "中文女")
            stream = await loop.run_in_executor(None, self._gen_sft, text, speaker)
        else:
            ref_audio, ref_text = self._resolve_ref(payload)
            stream = await loop.run_in_executor(None, self._gen_zero_shot, text, ref_text, ref_audio)

        return self._stream(stream), self.sample_rate, "application/octet-stream"

    # ── generation ─────────────────────────────────────────────

    def _gen_zero_shot(self, text, ref_text, ref_audio_path):
        import torch
        with torch.inference_mode():
            for out in self.model.inference_zero_shot(
                text, ref_text, ref_audio_path, stream=False
            ):
                yield out["tts_speech"]

    def _gen_sft(self, text, speaker):
        import torch
        with torch.inference_mode():
            for out in self.model.inference_sft(text, speaker, stream=False):
                yield out["tts_speech"]

    def _resolve_ref(self, payload):
        """Resolve reference audio from payload (path, base64, or file upload)."""
        ref_text = str(payload.get("ref_text") or "").strip()
        if not ref_text:
            raise ValueError("ref_text required")

        ref_b64 = payload.get("ref_audio_b64")
        if ref_b64:
            raw = base64.b64decode(ref_b64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(raw)
            return f.name, ref_text

        ref_path = payload.get("ref_audio")
        if not ref_path or not os.path.exists(str(ref_path)):
            raise ValueError("ref_audio or ref_audio_b64 is required for voice_clone mode")
        return ref_path, ref_text

    # ── internal ───────────────────────────────────────────────

    def _warmup(self):
        try:
            wav = os.path.join(self._repo_dir or "", "asset", "zero_shot_prompt.wav")
            if not os.path.isfile(wav):
                logger.warning("[CosyVoice] warmup skipped: ref audio not found")
                return
            ref = "希望你以后能够做的比我还好呦。"
            if self._is_cosyvoice3:
                ref = "You are a helpful assistant." + COSYVOICE3_END_PROMPT + ref
            logger.info("[CosyVoice] warming up...")
            for _ in self.model.inference_zero_shot("你好。", ref, wav, stream=False):
                pass
            logger.info("[CosyVoice] warmup done")
        except Exception as e:
            logger.warning("[CosyVoice] warmup failed (non-fatal): %s", e)

    @staticmethod
    def _stream(output):
        for t in output:
            a = t.cpu().numpy() if hasattr(t, "cpu") else np.asarray(t, dtype=np.float32)
            a = np.clip(a.reshape(-1), -1.0, 1.0)
            yield (a * 32767).astype(np.int16).tobytes()


# ── embedded singleton accessor ────────────────────────────────

def _get_or_create_embedded_backend() -> CosyVoiceBackend:
    global _EMBEDDED_BACKEND
    if _EMBEDDED_BACKEND is not None:
        return _EMBEDDED_BACKEND
    with _EMBEDDED_BACKEND_LOCK:
        if _EMBEDDED_BACKEND is not None:
            return _EMBEDDED_BACKEND
        args = SimpleNamespace(model_dir="", fp16=False, trt=False, vllm=False)
        backend = CosyVoiceBackend(args)
        backend.load()
        _EMBEDDED_BACKEND = backend
        return _EMBEDDED_BACKEND


# ── server-mode client (calls /v1/audio/speech) ───────────────

CHUNK_SIZE = 960


def _synthesize_via_server(text: str, ref_audio: str, ref_text: str, section: dict):
    """Call the standard /v1/audio/speech endpoint, yield int16 PCM chunks."""
    import requests

    host = str(section.get("host", "127.0.0.1"))
    port = int(section.get("port", 58099))
    addr = "127.0.0.1" if host == "0.0.0.0" else host
    timeout = int(section.get("request_timeout", 120))

    payload = {
        "input": text,
        "ref_text": ref_text,
        "ref_audio": ref_audio,
        "response_format": "pcm",
    }

    try:
        with requests.post(
            f"http://{addr}:{port}/v1/audio/speech",
            json=payload, stream=True, timeout=timeout,
        ) as res:
            if res.status_code != 200:
                logger.error("[CosyVoice-server] HTTP %s: %s", res.status_code, res.text[:500])
                raise RuntimeError(f"TTS server returned {res.status_code}")
            for chunk in res.iter_content(chunk_size=CHUNK_SIZE * 2):
                if chunk:
                    yield chunk
    except Exception:
        logger.exception("[CosyVoice-server] synthesis error")
        raise


# ── unified entrypoint ─────────────────────────────────────────

def synthesize_cosyvoice_stream(
    text: str,
    ref_audio: str,
    ref_text: str,
    config: dict,
    is_cancelled=None,
):
    """Synthesize text with voice cloning, yielding int16 PCM byte chunks.

    Auto-routes to embedded mode or server mode based on config:
      cosy_voice.standalone=true → call /v1/audio/speech endpoint
      cosy_voice.standalone=false → use in-process CosyVoiceBackend
    """
    section = config.get("cosy_voice", {}) if isinstance(config, dict) else {}
    standalone = bool(section.get("standalone", True))

    # ── server mode ──
    if standalone:
        yield from _synthesize_via_server(text, ref_audio, ref_text, section)
        return

    # ── embedded mode ──
    backend = _get_or_create_embedded_backend()
    if not ref_audio or not os.path.exists(str(ref_audio)):
        raise FileNotFoundError(f"ref_audio not found: {ref_audio}")

    import torch
    with torch.inference_mode():
        gen = backend._gen_zero_shot(text, ref_text, str(ref_audio))
    stream = backend._stream(gen)

    if is_cancelled:
        for chunk in stream:
            if is_cancelled():
                return
            yield chunk
    else:
        yield from stream
