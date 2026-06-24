import base64
import json
import pandas as pd
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="EMPTY"
)

df = pd.read_csv(
    "/mnt/data/hate-speech-bench-es/data/DIMEMEX/processed/dimemex24_unified_binary_benchmark.csv"
)

df = df.sample(10, random_state=42)

prompt_template = """
Eres un clasificador de discurso de odio en memes en español.

Clasifica el meme en una sola etiqueta:
- hate_speech
- no_hate_speech

Texto OCR del meme:
{ocr_text}

Usa la imagen y el texto. Devuelve exclusivamente JSON válido:
{{"label":"hate_speech"}}
o
{{"label":"no_hate_speech"}}
"""

rows = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    image_path = Path(row["image_path"])

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = prompt_template.format(
        ocr_text=str(row["text"])
    )

    try:
        response = client.chat.completions.create(
            model="BSC-LT/Salamandra-VL-7B-2512",
            temperature=0,
            max_tokens=80,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_b64}"
                            }
                        }
                    ]
                }
            ]
        )

        raw = response.choices[0].message.content.strip()

        try:
            parsed = json.loads(raw)
            pred = parsed.get("label")
        except Exception:
            pred = raw

    except Exception as e:
        raw = str(e)
        pred = "ERROR"

    rows.append({
        "meme_id": row["meme_id"],
        "gold": row["label_text"],
        "gold_binary": row["hate_binary"],
        "prediction": pred,
        "raw_output": raw
    })

out = pd.DataFrame(rows)
out.to_csv(
    "/mnt/data/hate-speech-bench-es/results/salamandra_vl_7b_dimemex_predictions.csv",
    index=False
)

print(out["prediction"].value_counts(dropna=False))
