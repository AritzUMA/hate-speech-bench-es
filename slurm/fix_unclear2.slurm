#!/bin/bash
#SBATCH --job-name=fix_unclear2
#SBATCH --output=/mnt/data/hate-speech-bench-es/logs/slurm/fix_unclear2_%j.out
#SBATCH --error=/mnt/data/hate-speech-bench-es/logs/slurm/fix_unclear2_%j.err
#SBATCH --partition=high
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00

source /mnt/data/hate-speech-bench-es/scripts/venv/bin/activate
export PYTHONUNBUFFERED=1
cd /mnt/data/hate-speech-bench-es
python3 scripts/fix_unclear2.py
