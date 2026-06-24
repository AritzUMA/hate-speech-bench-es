#!/bin/bash
#SBATCH --job-name=eval_mm_vllm
#SBATCH --output=/mnt/data/hate-speech-bench-es/logs/slurm/eval_mm_vllm_%j.out
#SBATCH --error=/mnt/data/hate-speech-bench-es/logs/slurm/eval_mm_vllm_%j.err
#SBATCH --partition=high
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=08:00:00

REPO="/mnt/data/hate-speech-bench-es"
PORT=${VLLM_PORT:-8001}
source "$REPO/scripts/venv/bin/activate"
cd "$REPO"

export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

echo "[vllm] Parando Ollama..."
sudo systemctl stop ollama
sleep 15

echo "[vllm] Arrancando servidor: $MODEL en puerto $PORT"
vllm serve "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --max-model-len 4096 \
    --dtype auto \
    --trust-remote-code \
    --generation-config vllm &
VLLM_PID=$!

echo "[vllm] Esperando que la API este lista..."
until python3 -c "
from openai import OpenAI
c = OpenAI(base_url='http://localhost:${PORT}/v1', api_key='x')
c.chat.completions.create(model='${MODEL}', messages=[{'role':'user','content':'hi'}], max_tokens=1)
" 2>/dev/null; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[vllm] ERROR: proceso vLLM muerto"
        sudo systemctl start ollama
        exit 1
    fi
    sleep 5
done
echo "[vllm] API lista"

echo "[eval] Lanzando: $MODEL x $DATASET"
python3 scripts/eval_model_multimodal.py --force \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --vllm-port "$PORT"

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null || true
echo "[vllm] Servidor detenido"
sudo systemctl start ollama
