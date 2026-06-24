#!/bin/bash
#SBATCH --job-name=eval_internvl2
#SBATCH --partition=high
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/hate-speech-bench-es/logs/slurm/eval_internvl2_%j.out
#SBATCH --error=/mnt/data/hate-speech-bench-es/logs/slurm/eval_internvl2_%j.err

MODEL="OpenGVLab/InternVL2-8B-AWQ"
DATASET="${DATASET:-multi3hate}"
PORT=8001

source /mnt/data/hate-speech-bench-es/scripts/venv/bin/activate

sudo systemctl stop ollama
sleep 15

vllm serve "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --max-model-len 4096 \
    --dtype auto \
    --trust-remote-code \
    --chat-template-content-format openai \
    --gpu-memory-utilization 0.90 &
VLLM_PID=$!

for i in $(seq 1 120); do
    if curl -s "http://localhost:${PORT}/v1/models" | grep -q "InternVL"; then
        echo "[vllm] Servidor listo"
        break
    fi
    sleep 5
done

cd /mnt/data/hate-speech-bench-es
python3 scripts/eval_model_multimodal.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --vllm-port "$PORT"

kill $VLLM_PID 2>/dev/null
sudo systemctl start ollama
