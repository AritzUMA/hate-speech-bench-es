#!/bin/bash
#SBATCH --job-name=debug_qwen3b
#SBATCH --partition=high
#SBATCH --priority=high
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/data/hate-speech-bench-es/logs/slurm/debug_qwen3b_%j.out
#SBATCH --error=/mnt/data/hate-speech-bench-es/logs/slurm/debug_qwen3b_%j.err

source /mnt/data/hate-speech-bench-es/scripts/venv/bin/activate

sudo systemctl stop ollama
sleep 5

vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
    --port 8001 \
    --host 0.0.0.0 \
    --max-model-len 4096 \
    --dtype auto \
    --trust-remote-code &
VLLM_PID=$!

for i in $(seq 1 120); do
    if curl -s "http://localhost:8001/v1/models" | grep -q "Qwen"; then
        echo "[vllm] Servidor listo"
        break
    fi
    sleep 5
done

python3 - << 'PYEOF'
from openai import OpenAI
import base64, time

client = OpenAI(base_url='http://localhost:8001/v1', api_key='x')

img_path = '/mnt/data/hate-speech-bench-es/data/Multi3Hate/data/memes/es/Scumbag-Catholic-Priest/275.jpg'
with open(img_path, 'rb') as f:
    img_b64 = base64.b64encode(f.read()).decode()

for attempt in range(1, 6):
    print(f"\n--- Intento {attempt} ---")
    try:
        r = client.chat.completions.create(
            model='Qwen/Qwen2.5-VL-3B-Instruct',
            messages=[
                {'role': 'system', 'content': 'Clasifica el meme. Responde SOLO con JSON: {"is_hate_speech": "Yes"/"No", "rationale": "..."}'},
                {'role': 'user', 'content': [
                    {'type': 'text', 'text': 'Texto: "Muestra tu odio por el sexo sano. ¡Hazte cura hoy mismo!". Clasifica este meme.'},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{img_b64}'}},
                ]},
            ],
            temperature=0,
            max_tokens=256,
        )
        raw = r.choices[0].message.content
        print(f"Respuesta: {repr(raw)}")
        if raw.strip():
            print("✅ Respuesta válida!")
            break
        else:
            print("⚠️ Respuesta vacía, reintentando...")
            time.sleep(3)
    except Exception as e:
        print(f"ERROR: {e}")
        time.sleep(3)
PYEOF

kill $VLLM_PID 2>/dev/null
sudo systemctl start ollama
