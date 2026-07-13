FROM vllm-fun-cosyvoice3:v1

ENV HF_ENDPOINT=https://hf-mirror.com
WORKDIR /home

RUN git clone --depth=1 https://github.com/FunAudioLLM/CosyVoice.git && \
    cd CosyVoice && \
    git clone --depth=1 https://github.com/shivammehta25/Matcha-TTS.git third_party/Matcha-TTS && \
    sed -i 's|gpu_memory_utilization=0.2)|gpu_memory_utilization=0.55, max_model_len=4096, compilation_config={"cudagraph_mode": "FULL_DECODE_ONLY"})|' cosyvoice/cli/model.py && \
    python3 -c "c=open('cosyvoice/vllm/cosyvoice2.py').read();c=c.replace('from typing import Optional','from typing import Optional, Union');open('cosyvoice/vllm/cosyvoice2.py','w').write(c)" && \
    pip install -q hyperpyyaml soundfile wget wetext pyworld conformer inflect hydra-core pyarrow ruamel.yaml networkx protobuf omegaconf gdown scipy torchaudio onnxruntime 2>&1 | tail -1 && \
    pip install -q lightning diffusers gradio x-transformers tensorboard openai-whisper matplotlib 2>&1 | tail -1

ENV PYTHONPATH=/home/CosyVoice:/home/CosyVoice/third_party/Matcha-TTS
ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

COPY sact_server.py /home/CosyVoice/server.py
EXPOSE 58002
CMD cd /home/CosyVoice && python3 server.py
