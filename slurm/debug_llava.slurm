#!/bin/bash
#SBATCH --job-name=debug_llava
#SBATCH --partition=high
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/data/hate-speech-bench-es/logs/slurm/debug_llava_%j.out
#SBATCH --error=/mnt/data/hate-speech-bench-es/logs/slurm/debug_llava_%j.err

source /mnt/data/hate-speech-bench-es/scripts/venv/bin/activate

python3 - << 'PYEOF'
from openai import OpenAI
import base64

client = OpenAI(base_url='http://localhost:8001/v1', api_key='x')

with open('/mnt/data/hate-speech-bench-es/data/Multi3Hate/data/memes/es/Scumbag-Catholic-Priest/275.jpg', 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()

try:
    r = client.chat.completions.create(
        model='llava-hf/llava-1.5-7b-hf',
        messages=[
            {'role': 'system', 'content': 'Clasifica el meme como hate o no_hate.'},
            {'role': 'user', 'content': [
                {'type': 'text', 'text': 'Texto: hola. Clasifica.'},
                {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{img_b64}'}},
            ]},
        ],
        temperature=0,
        max_tokens=50,
    )
    print('OK:', r.choices[0].message.content)
except Exception as e:
    print('ERROR:', type(e).__name__, str(e))
PYEOF
