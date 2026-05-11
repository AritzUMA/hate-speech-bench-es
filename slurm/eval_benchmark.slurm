#!/bin/bash
#SBATCH --job-name=eval_benchmark
#SBATCH --output=/mnt/data/hate-speech-bench-es/logs/slurm/eval_benchmark_%j.out
#SBATCH --error=/mnt/data/hate-speech-bench-es/logs/slurm/eval_benchmark_%j.err
#SBATCH --partition=low
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00

source /mnt/data/hate-speech-bench-es/scripts/venv/bin/activate
cd /mnt/data/hate-speech-bench-es

# MODEL y DATASET se pasan desde submit_benchmark.sh via --export
python3 scripts/eval_model.py --model "$MODEL" --dataset "$DATASET"
