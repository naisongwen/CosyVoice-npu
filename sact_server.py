import io, logging, os, sys, tempfile
import numpy as np, torch, torch_npu, soundfile as sf, uvicorn
from torch_npu.contrib import transfer_to_npu
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from contextlib import asynccontextmanager

MODEL_DIR = os.environ.get("COSYVOICE_MODEL_DIR", "/home/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B")
sys.path.append(os.environ.get("COSYVOICE_MATCHA_PATH", "/home/CosyVoice/third_party/Matcha-TTS"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sact")

_cosyvoice = None

def load_model():
    global _cosyvoice
    torch.npu.set_compile_mode(jit_compile=False)
    logger.info("Loading CosyVoice3+vLLM from %s ...", MODEL_DIR)
    from cosyvoice.cli.cosyvoice import AutoModel
    _cosyvoice = AutoModel(model_dir=MODEL_DIR, load_vllm=True)
    logger.info("Loaded, sr=%d", _cosyvoice.sample_rate)

    wav = "/home/CosyVoice/asset/zero_shot_prompt.wav"
    if os.path.isfile(wav):
        logger.info("Warmup...")
        for _ in _cosyvoice.inference_zero_shot(
            "你好。", "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。",
            wav, stream=False,
        ): pass
        torch.npu.synchronize()
        logger.info("Warmup done")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield

app = FastAPI(title="SACT CosyVoice3", lifespan=lifespan)

@app.get("/health")
def health():
    return {"status": "ok", "loaded": _cosyvoice is not None}

@app.post("/tts/zero_shot")
async def tts_zero_shot(
    tts_text: str = Form(...),
    prompt_text: str = Form(...),
    prompt_audio: UploadFile = File(...),
):
    if not _cosyvoice: raise HTTPException(503, "Not ready")
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(await prompt_audio.read())
        p = f.name
    try:
        a, sr = [], None
        for c in _cosyvoice.inference_zero_shot(tts_text, prompt_text, p, stream=False):
            if not sr: sr = _cosyvoice.sample_rate
            a.append(c["tts_speech"])
        if not a: raise HTTPException(500, "No audio")
        pcm = torch.cat(a, dim=-1).cpu().numpy().flatten()
        buf = io.BytesIO()
        sf.write(buf, np.clip(pcm,-1,1), sr or 24000, format="WAV", subtype="PCM_16")
        return Response(buf.getvalue(), media_type="audio/wav")
    except Exception as e:
        logger.exception("TTS failed")
        raise HTTPException(500, str(e))
    finally:
        try: os.unlink(p)
        except: pass

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=58002, log_level="info")
