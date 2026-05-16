#!/usr/bin/env bash
# Cambia entre backends Ollama y vLLM liberando la GPU correctamente.
# Uso:
#   bash scripts/switch_backend.sh ollama   # Para vLLM, arranca Ollama
#   bash scripts/switch_backend.sh vllm     # Para Ollama, arranca vLLM
#   bash scripts/switch_backend.sh status   # Muestra estado actual

set -euo pipefail

BACKEND=${1:-status}
OLLAMA_SERVICE="ollama"
GPU_FREE_THRESHOLD=14000  # MiB minimos para considerar GPU libre

gpu_free_mib() {
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1
}

ollama_running() {
    systemctl is-active --quiet "$OLLAMA_SERVICE" 2>/dev/null
}

vllm_running() {
    pgrep -f "vllm serve" > /dev/null 2>&1
}

wait_gpu_free() {
    echo "[switch] Esperando GPU libre (>${GPU_FREE_THRESHOLD}MiB)..."
    for i in $(seq 1 60); do
        FREE=$(gpu_free_mib)
        if [ "$FREE" -gt "$GPU_FREE_THRESHOLD" ]; then
            echo "[switch] GPU libre: ${FREE}MiB ✓"
            return 0
        fi
        echo "[switch]   intento $i/60 — libre: ${FREE}MiB, esperando 5s..."
        sleep 5
    done
    echo "[switch] ERROR: GPU no liberada tras 5 minutos (libre: $(gpu_free_mib)MiB)"
    return 1
}

case "$BACKEND" in

  status)
    echo "=== Estado del backend ==="
    echo "Ollama : $(ollama_running && echo 'ACTIVO' || echo 'parado')"
    echo "vLLM   : $(vllm_running && echo 'ACTIVO' || echo 'parado')"
    echo "GPU    : $(gpu_free_mib)MiB libres / 16376MiB total"
    ;;

  ollama)
    echo "[switch] → Cambiando a Ollama"

    # Para vLLM si está corriendo
    if vllm_running; then
        echo "[switch] Parando vLLM..."
        pkill -f "vllm serve" || true
        sleep 5
    fi

    # Arranca Ollama
    if ! ollama_running; then
        echo "[switch] Arrancando Ollama..."
        sudo systemctl start "$OLLAMA_SERVICE"
        sleep 5
    fi

    if ollama_running; then
        echo "[switch] Ollama activo ✓"
    else
        echo "[switch] ERROR: Ollama no arrancó"
        exit 1
    fi
    ;;

  vllm)
    echo "[switch] → Cambiando a vLLM"

    # Espera a que no haya jobs Ollama corriendo (estado R)
    echo "[switch] Esperando a que terminen los jobs Ollama en ejecucion..."
    while true; do
        RUNNING=$(squeue --me -h -o "%j %T" 2>/dev/null \
            | grep " R$" \
            | grep -v "vllm\|mm_vllm\|eval_mm_vllm" \
            | wc -l || true)
        if [ "$RUNNING" -eq 0 ]; then
            echo "[switch] No hay jobs Ollama corriendo"
            break
        fi
        echo "[switch]   $RUNNING jobs Ollama corriendo, esperando 30s..."
        sleep 30
    done

    # Descarga modelos de GPU antes de parar Ollama
    if ollama_running; then
        echo "[switch] Descargando modelos de GPU..."
        ollama ps 2>/dev/null | awk 'NR>1 {print $1}' \
            | xargs -I{} ollama stop {} 2>/dev/null || true
        sleep 10

        echo "[switch] Parando Ollama..."
        sudo systemctl stop "$OLLAMA_SERVICE"
    fi

    # Mata cualquier proceso residual de PyTorch/CUDA de Ollama
    echo "[switch] Limpiando procesos residuales..."
    pkill -f "ollama" 2>/dev/null || true
    sleep 5

    # Espera a que la GPU quede realmente libre
    wait_gpu_free

    echo "[switch] Listo para vLLM ✓ — GPU libre: $(gpu_free_mib)MiB"
    ;;

  *)
    echo "Uso: $0 [ollama|vllm|status]"
    exit 1
    ;;
esac
