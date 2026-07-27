#!/bin/bash
#SBATCH --job-name=test_fix
#SBATCH --output=/mnt/data/hate-speech-bench-es/logs/slurm/test_fix_%j.out
#SBATCH --error=/mnt/data/hate-speech-bench-es/logs/slurm/test_fix_%j.err
#SBATCH --partition=high
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00

source /mnt/data/hate-speech-bench-es/scripts/venv/bin/activate
export PYTHONUNBUFFERED=1
cd /mnt/data/hate-speech-bench-es
python3 scripts/test_fix_unclear.py
