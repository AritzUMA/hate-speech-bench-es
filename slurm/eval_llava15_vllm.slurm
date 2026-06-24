#!/bin/bash
#SBATCH --job-name=eval_llava15
#SBATCH --partition=high
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/data/hate-speech-bench-es/logs/slurm/eval_llava15_%j.out
#SBATCH --error=/mnt/data/hate-speech-bench-es/logs/slurm/eval_llava15_%j.err

MODEL="llava-hf/llava-1.5-7b-hf"
DATASET="${DATASET:-multi3hate}"
PORT=8001

source /mnt/data/hate-speech-bench-es/scripts/venv/bin/activate

echo "[vllm] Parando Ollama..."
sudo systemctl stop ollama
sleep 15

echo "[vllm] Arrancando servidor: $MODEL en puerto $PORT"
vllm serve "$MODEL" \
    --chat-template /mnt/data/hate-speech-bench-es/scripts/template_llava.jinja \
    --port "$PORT" \
    --host 0.0.0.0 \
    --max-model-len 1024 \
    --dtype auto \
    --trust-remote-code \
    --gpu-memory-utilization 0.80 \
    --enforce-eager &
VLLM_PID=$!

echo "[vllm] Esperando que el modelo cargue..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:${PORT}/v1/models" | grep -q "$MODEL"; then
        echo "[vllm] Servidor listo (${i}*5s)"
        break
    fi
    sleep 5
done

if ! curl -s "http://localhost:${PORT}/v1/models" | grep -q "$MODEL"; then
    echo "[vllm] ERROR: servidor no arrancó"
    kill $VLLM_PID 2>/dev/null
    sudo systemctl start ollama
    exit 1
fi

echo "[eval] Lanzando: $MODEL x $DATASET"
cd /mnt/data/hate-speech-bench-es
python3 scripts/eval_model_multimodal.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --vllm-port "$PORT"

kill $VLLM_PID 2>/dev/null
echo "[vllm] Servidor detenido"
sudo systemctl start ollama
echo "[vllm] Ollama reiniciado"
