#!/bin/bash
#SBATCH --job-name=eval_multimodal
#SBATCH --output=/mnt/data/hate-speech-bench-es/logs/slurm/eval_multimodal_%j.out
#SBATCH --error=/mnt/data/hate-speech-bench-es/logs/slurm/eval_multimodal_%j.err
#SBATCH --partition=high
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00

source /mnt/data/hate-speech-bench-es/scripts/venv/bin/activate
cd /mnt/data/hate-speech-bench-es

# MODEL y DATASET se pasan desde submit_multimodal.sh via --export
python3 scripts/eval_model_multimodal.py --model "$MODEL" --dataset "$DATASET"
