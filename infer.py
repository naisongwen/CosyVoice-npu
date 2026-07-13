import sys
import torch
import torchaudio
from cosyvoice.cli.cosyvoice import AutoModel
import torch_npu
from torch_npu.contrib import transfer_to_npu

import os

if __name__ == "__main__":
    torch_npu.npu.set_compile_mode(jit_compile=False)
    #os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '2'
    print("Torch npu available:", torch_npu.npu.is_available())
    torch_npu.npu.set_compile_mode(jit_compile=False)

    # 添加路径
    sys.path.append('third_party/Matcha-TTS')

    # 选择设备 (如果可用则使用 NPU，否则使用 CPU)
    device = "npu" if torch_npu.npu.is_available() else "cpu"
    print(f"Using device: {device}")

    # 初始化模型并确保使用 NPU
    cosyvoice = AutoModel(model_dir='./pretrained_models/Fun-CosyVoice3-0.5B/', load_vllm = True)

    # zero_shot 使用
    for i, j in enumerate(cosyvoice.inference_zero_shot(
        '八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。',
        'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
        './asset/zero_shot_prompt.wav', stream=False)):

        # 保存时转到 CPU
        torchaudio.save(f'zero_shot_{i}.wav', j['tts_speech'].cpu(), cosyvoice.sample_rate)

    # fine-grained control 使用
    for i, j in enumerate(cosyvoice.inference_cross_lingual(
        'You are a helpful assistant.<|endofprompt|>因为他们那一辈人[breath]在乡里面住的要习惯一点，[breath]邻居都很活络，[breath]嗯，都很熟悉.',
        './asset/zero_shot_prompt.wav', stream=False)):

        torchaudio.save(f'fine_grained_control_{i}.wav', j['tts_speech'].cpu(), cosyvoice.sample_rate)

    # instruct 使用
    for i, j in enumerate(cosyvoice.inference_instruct2(
        '好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。',
        'You are a helpful assistant. 请用广东话表达。<|endofprompt|>',
        './asset/zero_shot_prompt.wav', stream=False)):

        torchaudio.save(f'instruct_{i}.wav', j['tts_speech'].cpu(), cosyvoice.sample_rate)

    for i, j in enumerate(cosyvoice.inference_instruct2(
        '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐，笑容如花儿般绽放。',
        'You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>',
        './asset/zero_shot_prompt.wav', stream=False)):

        torchaudio.save(f'instruct2_{i}.wav', j['tts_speech'].cpu(), cosyvoice.sample_rate)  # 改名避免覆盖

    # hotfix 使用
    for i, j in enumerate(cosyvoice.inference_zero_shot(
        '高管也通过电话、短信、微信等方式对报道[j][ǐ]予好评。',
        'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
        './asset/zero_shot_prompt.wav', stream=False)):

        torchaudio.save(f'hotfix_{i}.wav', j['tts_speech'].cpu(), cosyvoice.sample_rate)
