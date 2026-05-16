#!/usr/bin/env bash
# Relanza via SLURM todos los runs con cobertura de prediccion por debajo
# del umbral minimo (default: 100%).
#
# Uso: bash scripts/retry_failed.sh
#      bash scripts/retry_failed.sh --threshold 0.90

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$REPO_ROOT/logs/slurm"
mkdir -p "$LOG_DIR"

THRESHOLD=1.0

while [[ $# -gt 0 ]]; do
  case $1 in
    --threshold) THRESHOLD="$2"; shift 2 ;;
    *) echo "Argumento desconocido: $1"; exit 1 ;;
  esac
done

echo "Umbral minimo de cobertura: ${THRESHOLD}"
echo ""

TMPPY=$(mktemp /tmp/retry_XXXXXX.py)
cat > "$TMPPY" << PYEOF
import json, sys
from pathlib import Path

repo      = Path(sys.argv[1])
threshold = float(sys.argv[2])
runs_dir  = repo / "results" / "runs"

for f in sorted(runs_dir.glob("*.json")):
    try:
        r = json.loads(f.read_text())
        n   = r.get("n_instances", 0)
        sin = r.get("n_unclear", r.get("n_sin_pred", 0))
        if n == 0:
            continue
        coverage = (n - sin) / n
        if coverage < threshold:
            print(f"{r['model']}|{r['dataset']}|{round(coverage*100,1)}|{f.name}")
    except Exception as e:
        print(f"  [warn] skip {f.name}: {e}", file=sys.stderr)
PYEOF

PAIRS=$(python3 "$TMPPY" "$REPO_ROOT" "$THRESHOLD")
rm "$TMPPY"

if [[ -z "$PAIRS" ]]; then
  echo "Todos los runs tienen cobertura al 100% — nada que relanzar."
  exit 0
fi

echo "Runs con cobertura insuficiente:"
echo ""
while IFS='|' read -r MODEL DATASET COVERAGE FILE; do
  echo "  ${COVERAGE}%  ${MODEL} x ${DATASET}"
done <<< "$PAIRS"
echo ""

JOB_IDS=()

while IFS='|' read -r MODEL DATASET COVERAGE FILE; do
  SAFE=$(echo "$MODEL" | tr ':/' '--')
  JOB_NAME="retry_${SAFE}_${DATASET}"

  # Borra el run anterior
  rm -f "$REPO_ROOT/results/runs/$FILE"
  echo "  [removed] $FILE"

  JOB_ID=$(sbatch \
    --parsable \
    --job-name="$JOB_NAME" \
    --export="MODEL=$MODEL,DATASET=$DATASET" \
    "$REPO_ROOT/slurm/eval_benchmark.slurm")

  echo "  submitted  $JOB_NAME -> job $JOB_ID  (cobertura anterior: ${COVERAGE}%)"
  JOB_IDS+=("$JOB_ID")

done <<< "$PAIRS"

echo ""
echo "Total: ${#JOB_IDS[@]} jobs relanzados"
