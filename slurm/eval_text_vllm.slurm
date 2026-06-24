#!/bin/bash
#SBATCH --job-name=eval_txt_vllm
#SBATCH --output=/mnt/data/hate-speech-bench-es/logs/slurm/eval_txt_vllm_%j.out
#SBATCH --error=/mnt/data/hate-speech-bench-es/logs/slurm/eval_txt_vllm_%j.err
#SBATCH --partition=high
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=08:00:00

REPO="/mnt/data/hate-speech-bench-es"
PORT=${VLLM_PORT:-8001}

source "$REPO/scripts/venv/bin/activate"
cd "$REPO"

export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HOME=/mnt/data/.cache/huggingface

echo "[vllm] Parando Ollama..."
sudo systemctl stop ollama
sleep 15

echo "[vllm] Arrancando servidor: $MODEL en puerto $PORT"
vllm serve "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --max-model-len 4096 \
    --dtype auto \
    --trust-remote-code &

VLLM_PID=$!

echo "[vllm] Esperando que el modelo cargue..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:${PORT}/v1/models" | grep -q "$MODEL"; then
        echo "[vllm] Modelo cargado y listo (${i}*5s)"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[vllm] ERROR: proceso vLLM muerto"
        sudo systemctl start ollama
        exit 1
    fi
    sleep 5
done

echo "[eval] Lanzando: $MODEL x $DATASET"
python3 scripts/eval_model_vllm.py --force \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --vllm-port "$PORT"

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null || true
echo "[vllm] Servidor detenido"

# Elimina cache del modelo
SAFE_MODEL=$(echo "$MODEL" | tr '/' '--')
rm -rf /mnt/data/.cache/huggingface/hub/models--${SAFE_MODEL}
echo "[cleanup] Cache eliminada"

sudo systemctl start ollama
