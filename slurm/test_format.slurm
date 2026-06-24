#!/bin/bash
#SBATCH --job-name=test_fmt
#SBATCH --partition=high
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/data/hate-speech-bench-es/logs/slurm/test_format_%j.out

IMG=$(find /mnt/data/hate-speech-bench-es/data/Multi3Hate/data/memes/es -name "*.jpg" | head -1)

source /mnt/data/hate-speech-bench-es/scripts/venv/bin/activate

python3 -c "
import ollama, base64
img = base64.b64encode(open('$IMG','rb').read()).decode()

for model in ['gemma4:e4b', 'qwen3-vl:4b']:
    print(f'=== {model} ===')
    try:
        r = ollama.chat(model=model, messages=[{'role':'user','content':'Describe this image','images':[img]}], options={'temperature':0}, format={'type':'object','properties':{'description':{'type':'string'}},'required':['description']})
        print(f'CON format: {repr(r[\"message\"][\"content\"][:200])}')
    except Exception as e:
        print(f'CON format ERROR: {e}')
    try:
        r2 = ollama.chat(model=model, messages=[{'role':'user','content':'Describe this image','images':[img]}], options={'temperature':0})
        print(f'SIN format: {repr(r2[\"message\"][\"content\"][:200])}')
    except Exception as e:
        print(f'SIN format ERROR: {e}')
    print()
"
