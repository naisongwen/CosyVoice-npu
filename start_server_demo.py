import sys
import logging
from io import BytesIO
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from cosyvoice.cli.cosyvoice import AutoModel
from pydantic import BaseModel
from typing import Optional
import torch
import torchaudio
import torch_npu
from torch_npu.contrib import transfer_to_npu
from cosyvoice.cli.cosyvoice import AutoModel
"""
Fun-CosyVoice3-0.5B 服务化脚本
启动前请执行：
export VLLM_WORKER_MULTIPROC_METHOD=spawn
"""
# 设置模型路径
MODEL_PATH='pretrained_models/Fun-CosyVoice3-0.5B'
MATCHA_TTS_PATH='third_party/Matcha-TTS'
SERVER_PORT=8002
# Uvicorn 的并发进程数
WORKERS=2   

# 全局模型对象，必须在方法内初始化
cosyvoice = None

def load_model():
    
    sys.path.append(MATCHA_TTS_PATH)
    logging.info("Loading CosyVoice model...")
    model = AutoModel(model_dir=MODEL_PATH, load_vllm=True)
    logging.info("CosyVoice model loaded successfully.")
    return model

app = FastAPI(title="CosyVoice3 Service", description="API for CosyVoice3 TTS model.")


@app.on_event("startup")
def startup_event():
    global cosyvoice
    try:
        torch_npu.npu.set_compile_mode(jit_compile=False)
        cosyvoice = load_model()
    except Exception as e:
        logging.error(f"Failed to load CosyVoice model during startup: {e}")
        raise RuntimeError(f"Service startup failed due to model loading error: {e}")


# --- 定义请求体模型 ---
class ZeroShotRequest(BaseModel):
    tts_text: str
    prompt_text: str
    prompt_wav_path: Optional[str] = None # 如果使用上传的音频文件，可以移除此字段

class CrossLingualRequest(BaseModel):
    tts_text: str
    prompt_wav_path: Optional[str] = None

class InstructRequest(BaseModel):
    tts_text: str
    instruct_text: str
    prompt_wav_path: Optional[str] = None

@app.post("/tts/zero_shot")
async def tts_zero_shot(tts_text: str = Form(...),
                        prompt_text: str = Form(...),
                        prompt_audio: UploadFile = File(None)):
    """
    Zero-shot TTS inference.
    - tts_text: The text to synthesize.
    - prompt_text: The prompt text.
    - prompt_audio: Optional audio file to provide voice characteristics.
    """
    try:
        # 处理上传的音频文件
        prompt_wav_path = None
        if prompt_audio:
            # 将上传的文件保存到临时路径或直接加载到内存
            import tempfile
            import os
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(prompt_audio.filename)[1]) as tmp_file:
                content = await prompt_audio.read()
                tmp_file.write(content)
                prompt_wav_path = tmp_file.name

        # --- 执行推理 ---
        # 注意：这里stream=False, 返回完整的音频
        # 由于inference_zero_shot返回的是生成器，我们需要收集所有片段
        full_audio = []
        sample_rate = None
        for chunk in cosyvoice.inference_zero_shot(tts_text, prompt_text, prompt_wav_path, stream=False):
            if sample_rate is None:
                sample_rate = cosyvoice.sample_rate
            full_audio.append(chunk['tts_speech'])

        if not full_audio:
            raise HTTPException(status_code=500, detail="No audio generated")

        # 拼接音频片段
        final_audio = torch.cat(full_audio, dim=-1)

        # 将音频数据转换为字节流
        buffer = BytesIO()
        torchaudio.save(buffer, final_audio, sample_rate, format='wav')
        buffer.seek(0)
        wav_data = buffer.getvalue()

        # 清理临时文件
        if prompt_wav_path:
            os.unlink(prompt_wav_path)

        # 返回音频文件
        return StreamingResponse(BytesIO(wav_data), media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=output.wav"}    )

    except Exception as e:
        logging.error(f"Error in zero-shot TTS: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/cross_lingual")
async def tts_cross_lingual(tts_text: str = Form(...),
                            prompt_audio: UploadFile = File(...)):
    """
    Cross-lingual TTS inference.
    - tts_text: The text to synthesize (can be in a different language).
    - prompt_audio: Audio file to provide voice characteristics.
    """
    try:
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(prompt_audio.filename)[1]) as tmp_file:
            content = await prompt_audio.read()
            tmp_file.write(content)
            prompt_wav_path = tmp_file.name

        full_audio = []
        sample_rate = None
        for chunk in cosyvoice.inference_cross_lingual(tts_text, prompt_wav_path, stream=False):
            if sample_rate is None:
                sample_rate = cosyvoice.sample_rate
            full_audio.append(chunk['tts_speech'])

        if not full_audio:
            raise HTTPException(status_code=500, detail="No audio generated")

        final_audio = torch.cat(full_audio, dim=-1)

        buffer = BytesIO()
        torchaudio.save(buffer, final_audio, sample_rate, format='wav')
        buffer.seek(0)
        wav_data = buffer.getvalue()

        os.unlink(prompt_wav_path)

        return StreamingResponse(BytesIO(wav_data), media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=output.wav"}    )

    except Exception as e:
        logging.error(f"Error in cross-lingual TTS: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tts/instruct")
async def tts_instruct(tts_text: str = Form(...),
                       instruct_text: str = Form(...),
                       prompt_audio: UploadFile = File(...)):
    """
    Instruct-based TTS inference.
    - tts_text: The text to synthesize.
    - instruct_text: Instruction for style, emotion, etc.
    - prompt_audio: Audio file to provide voice characteristics.
    """
    try:
        import tempfile
        import os
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(prompt_audio.filename)[1]) as tmp_file:
            content = await prompt_audio.read()
            tmp_file.write(content)
            prompt_wav_path = tmp_file.name

        full_audio = []
        sample_rate = None
        for chunk in cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_wav_path, stream=False):
            if sample_rate is None:
                sample_rate = cosyvoice.sample_rate
            full_audio.append(chunk['tts_speech'])

        if not full_audio:
            raise HTTPException(status_code=500, detail="No audio generated")

        final_audio = torch.cat(full_audio, dim=-1)

        buffer = BytesIO()
        torchaudio.save(buffer, final_audio, sample_rate, format='wav')
        buffer.seek(0)
        wav_data = buffer.getvalue()

        os.unlink(prompt_wav_path)

        return StreamingResponse(BytesIO(wav_data), media_type="audio/wav", headers={"Content-Disposition": "attachment; filename=output.wav"}    )

    except Exception as e:
        logging.error(f"Error in instruct TTS: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- 可选：健康检查接口 ---
@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    # 启动服务
    # workers=1: 默认单进程，如果需要并发处理，可以考虑增加workers
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT, workers=WORKERS)


"""
test:
curl -X POST "http://127.0.0.1:8002/tts/zero_shot" \
 -H "Content-Type: multipart/form-data" \
 -F "tts_text=八百标兵奔北坡，北坡炮兵并排跑。" \
 -F "prompt_text=You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。" \
 -F "prompt_audio=@./asset/zero_shot_prompt.wav" \
 --output output.wav

"""