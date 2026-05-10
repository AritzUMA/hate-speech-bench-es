#!/usr/bin/env bash
# Lanza un job SLURM por cada combinacion modelo vision x dataset multimodal
# que aun no tenga resultado en results/runs/.
#
# Uso: bash scripts/submit_multimodal.sh
# Uso (solo un dataset): bash scripts/submit_multimodal.sh --dataset multi3hate
# Uso (solo un modelo):  bash scripts/submit_multimodal.sh --model llama3.2-vision:latest

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs/slurm"
mkdir -p "$LOG_DIR"

# Args opcionales
FILTER_DATASET=""
FILTER_MODEL=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --dataset) FILTER_DATASET="$2"; shift 2 ;;
    --model)   FILTER_MODEL="$2";   shift 2 ;;
    *) echo "Argumento desconocido: $1"; exit 1 ;;
  esac
done

# Pendientes: modelos vision x datasets multimodal sin resultado
TMPPY=$(mktemp /tmp/pending_mm_XXXXXX.py)
cat > "$TMPPY" << PYEOF
import json, yaml, sys
from pathlib import Path

repo           = Path(sys.argv[1])
filter_dataset = sys.argv[2]
filter_model   = sys.argv[3]

with open(repo / "registry/models.yaml") as f:
    all_models = [
        m["ollama_name"] for m in yaml.safe_load(f)["models"]
        if m.get("modality") == "image+text"
    ]

with open(repo / "registry/datasets.yaml") as f:
    all_datasets = [
        d["name"] for d in yaml.safe_load(f)["datasets"]
        if d.get("modality") == "image+text"
    ]

if filter_dataset:
    all_datasets = [d for d in all_datasets if d == filter_dataset]
if filter_model:
    all_models = [m for m in all_models if m == filter_model]

done = set()
runs_dir = repo / "results" / "runs"
if runs_dir.exists():
    for f in runs_dir.glob("*.json"):
        try:
            d = json.loads(f.read_text())
            if d.get("model") and d.get("dataset"):
                done.add((d["model"], d["dataset"]))
        except Exception:
            pass

for model in all_models:
    for dataset in all_datasets:
        if (model, dataset) not in done:
            print(f"{model}|{dataset}")
PYEOF

PAIRS=$(python3 "$TMPPY" "$REPO_ROOT" "$FILTER_DATASET" "$FILTER_MODEL")
rm "$TMPPY"

if [[ -z "$PAIRS" ]]; then
  echo "Todas las combinaciones vision x multimodal ya estan evaluadas."
  exit 0
fi

N_PAIRS=$(echo "$PAIRS" | wc -l)
echo "Combinaciones pendientes: $N_PAIRS"
echo ""

JOB_IDS=()

while IFS='|' read -r MODEL DATASET; do
  SAFE=$(echo "$MODEL" | tr ':/' '--')
  JOB_NAME="mm_${SAFE}_${DATASET}"

  JOB_ID=$(sbatch \
    --parsable \
    --job-name="$JOB_NAME" \
    --output="$LOG_DIR/${JOB_NAME}_%j.out" \
    --error="$LOG_DIR/${JOB_NAME}_%j.err" \
    --export="MODEL=$MODEL,DATASET=$DATASET" \
    "$REPO_ROOT/slurm/eval_multimodal.slurm")

  echo "  submitted  $JOB_NAME -> job $JOB_ID"
  JOB_IDS+=("$JOB_ID")

done <<< "$PAIRS"

echo ""
echo "Total: ${#JOB_IDS[@]} jobs lanzados"
