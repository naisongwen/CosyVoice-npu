---
license: apache-2.0
---
### Fun-CosyVoice3-0.5B-2512 vllm-ascend 推理

#### 环境配置
| 环境配置 |               配置说明                |
| :------: | :-----------------------------------: |
| 硬件配置 |          Atlas A2 910B3/4(64G)          |
| 驱动版本 |                25.2.3                 |
| CANN版本 |                 8.3                 |
| 推理框架 |              vllm-ascend              |
| 部署方式 |               1卡 部署                |
| 本镜像架构 |               ARM                |



#### 部署步骤
提前下载本仓所有文件到本地

1.加载镜像
```
docker load -i vllm-fun-cosyvoice3-0.5B-v1.tar.gz  
```
2.启动容器
```
docker run -itd -u root --ipc=host --net=host --name=vllm_fun_cosyvoice3 --privileged=true \
-v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
-v /usr/local/dcmi:/usr/local/dcmi \
-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
-v /usr/local/sbin:/usr/local/sbin \
-v /home:/home \
--shm-size=10g \
vllm-fun-cosyvoice3:v1 \
/bin/bash
```
3.假定以/home/xxx为工作目录，进入容器
```
docker exec -it vllm_fun_cosyvoice3 bash
cd /home/xxx
```

4.下载代码
```
git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
```
5.打上代码补丁  
将本仓中的 cosyvoice3.patch、infer.py、download_weight.py 拷贝到  /home/xxx/CosyVoice
```
cd /home/xxx/CosyVoice
git apply cosyvoice3.patch
```
6.下载权重
```
python download_weight.py
```
无法联网时，可手工从如下路径下载权重
https://modelscope.cn/models/FunAudioLLM/Fun-CosyVoice3-0.5B-2512/files

7.整体目录结构如下
```
/home/xxx/CosyVoice
--infer.py
--cosyvoice3.patch
--download_weight.py 
--pretrained_models /  #权重路径
      -- Fun-CosyVoice3-0.5B/
      --CosyVoice-ttsfrd/
-- 其他文件
```
8.执行推理测试
```
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python infer.py
```
#### 测试效果 RTF≈0.27
![](./fun-cosyvoice3.PNG)


#### 服务化推理
本仓 start_server_demo.py 是一个使用fastapi简单封装的服务化脚本  
配置好脚本中的关键参数，即可启动服务  
MODEL_PATH='pretrained_models/Fun-CosyVoice3-0.5B'  
MATCHA_TTS_PATH='third_party/Matcha-TTS'  
SERVER_PORT=8002  
WORKERS=2 # Uvicorn 的并发进程数

- 启动服务
```
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python start_server_demo.py
```

- curl 测试
```
curl -X POST "http://127.0.0.1:8002/tts/zero_shot" \
 -H "Content-Type: multipart/form-data" \
 -F "tts_text=八百标兵奔北坡，北坡炮兵并排跑。" \
 -F "prompt_text=You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。" \
 -F "prompt_audio=@./asset/zero_shot_prompt.wav" \
 --output output.wav
```
预期输出
![](./curl_demo.PNG)