#!/bin/bash
set -e
# Source CANN environment
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
source /usr/local/Ascend/nnal/atb/set_env.sh 2>/dev/null || true
export PYTHONPATH="/workspace/CosyVoice:/workspace/CosyVoice/third_party/Matcha-TTS:$PYTHONPATH"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-1}"
export TORCHAUDIO_USE_SOUNDFILE_LEGACY_INTERFACE="${TORCHAUDIO_USE_SOUNDFILE_LEGACY_INTERFACE:-1}"
exec "$@"
