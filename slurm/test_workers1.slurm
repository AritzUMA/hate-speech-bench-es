#!/bin/bash
#SBATCH --job-name=test_w1
#SBATCH --partition=high
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/data/hate-speech-bench-es/logs/slurm/test_w1_%j.out
#SBATCH --error=/mnt/data/hate-speech-bench-es/logs/slurm/test_w1_%j.err

source /mnt/data/hate-speech-bench-es/scripts/venv/bin/activate
cd /mnt/data/hate-speech-bench-es

python3 scripts/eval_model_multimodal.py --force \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --workers 1
