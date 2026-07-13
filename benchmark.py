#!/usr/bin/env python3
"""CosyVoice3 Ascend NPU RTF Benchmark"""
import os, sys, time, torch, torch_npu
from torch_npu.contrib import transfer_to_npu
import soundfile as sf, torchaudio

if __name__ == "__main__":
    torch.npu.set_compile_mode(jit_compile=False)
    def lw(w,**kw):
        d,sr=sf.read(w,dtype="float32")
        if d.ndim==1:d=d.reshape(1,-1)
        else:d=d.T
        return torch.from_numpy(d.copy()),sr
    torchaudio.load=lw

    sys.path.insert(0,"/home/CosyVoice")
    sys.path.insert(0,"/home/CosyVoice/third_party/Matcha-TTS")
    from cosyvoice.cli.cosyvoice import AutoModel

    MODEL=os.environ.get("MODEL_DIR","pretrained_models/Fun-CosyVoice3-0.5B")
    REF_WAV=os.environ.get("REF_WAV","asset/zero_shot_prompt.wav")
    REF_TEXT="You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"

    print("Loading CosyVoice3+vLLM...",flush=True)
    model=AutoModel(model_dir=MODEL,load_vllm=True)

    for i in range(3):
        for _ in model.inference_zero_shot("你好。",REF_TEXT,REF_WAV,stream=False):pass
        torch.npu.synchronize()

    texts=[
        "你好，今天天气真不错。",
        "欢迎使用实时语音合成系统进行测试。",
        "八百标兵奔北坡，北坡炮兵并排跑。",
        "床前明月光，疑是地上霜。举头望明月，低头思故乡。",
    ]
    for text in texts:
        t0=time.perf_counter()
        for out in model.inference_zero_shot(text,REF_TEXT,REF_WAV,stream=False):
            audio=out["tts_speech"]
        torch.npu.synchronize()
        dt=time.perf_counter()-t0
        dur=audio.shape[1]/model.sample_rate
        rtf=dt/dur if dur>0 else 0
        print(f"  Audio={dur:.2f}s Time={dt:.1f}s RTF={rtf:.3f} {'OK' if rtf<1 else 'SLOW'}",flush=True)
