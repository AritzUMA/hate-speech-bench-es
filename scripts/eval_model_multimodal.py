#!/usr/bin/env python3
"""
Evalua un modelo vision de Ollama sobre cualquier dataset multimodal del benchmark.
Lee la configuracion de registry/datasets.yaml y registry/models.yaml.

Uso:
    python3 scripts/eval_model_multimodal.py --model llama3.2-vision:latest --dataset multi3hate
    python3 scripts/eval_model_multimodal.py --model gemma4:e2b --dataset multi3hate --dry-run
    python3 scripts/eval_model_multimodal.py --model moondream:latest --dataset multi3hate --force
"""
import argparse
import base64
import json
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import ollama
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score,
)
from tqdm import tqdm

# Rutas
REPO_ROOT  = Path(__file__).parent.parent
SAVE_EVERY = 20
N_WORKERS  = 4   # vision es mas pesado que texto

# Prompt
SYSTEM_PROMPT = """Eres un clasificador de memes en espanol.
Analiza la imagen y el texto del meme y determina si contiene discurso de odio.
Responde SOLO con un objeto JSON exactamente en este formato:
{
  "is_hate_speech": "Yes" | "No" | "Unclear",
  "rationale": "Breve explicacion (1-2 frases)"
}
No incluyas texto adicional, ni comentarios, ni markdown. SOLO JSON puro."""


# Registry

def get_dataset_cfg(dataset_name: str) -> dict:
    with open(REPO_ROOT / "registry" / "datasets.yaml") as f:
        datasets = yaml.safe_load(f)["datasets"]
    for d in datasets:
        if d["name"] == dataset_name:
            if d.get("modality") != "image+text":
                raise ValueError(
                    f"Dataset '{dataset_name}' no es multimodal (modality={d.get('modality')}). "
                    f"Usa eval_model.py para datasets de texto."
                )
            return d
    avail = [d["name"] for d in datasets if d.get("modality") == "image+text"]
    raise ValueError(f"Dataset '{dataset_name}' no encontrado. Disponibles image+text: {avail}")


def get_model_meta(model_name: str) -> dict:
    with open(REPO_ROOT / "registry" / "models.yaml") as f:
        models = yaml.safe_load(f)["models"]
    for m in models:
        if m["ollama_name"] == model_name:
            return m
    return {"ollama_name": model_name, "family": "unknown",
            "params": "unknown", "size_gb": None, "modality": "image+text"}


def get_evaluated(runs_dir: Path, dataset_name: str) -> set:
    evaluated = set()
    for f in runs_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            if data.get("dataset") == dataset_name:
                evaluated.add(data["model"])
        except Exception:
            pass
    return evaluated


# Loader

def load_dataset(cfg: dict) -> list:
    path = REPO_ROOT / cfg["path"]
    df   = pd.read_csv(path)
    text_col  = cfg["text_col"]
    image_col = cfg["image_col"]
    label_col = cfg["label_col"]
    df = df[[text_col, image_col, label_col]].dropna()
    df.columns = ["text", "image_path", "label"]
    df["label"] = df["label"].str.strip()
    records = df.to_dict(orient="records")
    valid = [r for r in records if r["label"] in ("hate", "no_hate")]
    if len(valid) < len(records):
        print(f"  [warn] {len(records) - len(valid)} filas con label inesperado excluidas")
    return valid


# Clasificador

def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def classify_meme(model: str, text: str, image_path: str) -> str:
    """
    Clasifica un meme (imagen + texto).
    Devuelve 'hate', 'no_hate' o 'unclear'.
    Un solo intento — si falla devuelve 'unclear'.
    """
    try:
        img_b64    = image_to_base64(image_path)
        caption    = text.replace("<sep>", "\n").strip()
        user_content = f'Texto del meme:\n"{caption}"\n\nClasifica este meme.'

        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role":    "user",
                    "content": user_content,
                    "images":  [img_b64],
                },
            ],
            options={"temperature": 0},
        )

        raw = response["message"]["content"].strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        if not raw.startswith("{"):
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end > start:
                raw = raw[start:end + 1]

        clf = json.loads(raw)
        val = str(clf.get("is_hate_speech", "")).strip().lower()

        if val in ("yes", "si", "si"):
            return "hate"
        if val in ("no", "unclear"):
            return "no_hate" if val == "no" else "unclear"
        return "unclear"

    except Exception:
        return "unclear"


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Evalua un modelo vision sobre un dataset multimodal del benchmark")
    parser.add_argument("--model",   required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Clasifica solo los primeros 5 memes sin guardar")
    parser.add_argument("--force",   action="store_true",
                        help="Evalua aunque ya exista resultado")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    args = parser.parse_args()

    runs_dir = REPO_ROOT / "results" / "runs"
    pred_dir = REPO_ROOT / "results" / "predictions"
    runs_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    ts         = datetime.now(timezone.utc)
    safe_model = args.model.replace(":", "-").replace("/", "-")

    # Comprobar si ya evaluado
    if not args.force and not args.dry_run:
        evaluated = get_evaluated(runs_dir, args.dataset)
        if args.model in evaluated:
            print(f"[eval] {args.model} ya evaluado para {args.dataset}.")
            print("       Usa --force para repetir.")
            return

    # Cargar dataset y metadata
    ds_cfg     = get_dataset_cfg(args.dataset)
    model_meta = get_model_meta(args.model)
    test_set   = load_dataset(ds_cfg)

    if args.dry_run:
        test_set = test_set[:5]

    print(f"[eval] modelo    : {args.model}")
    print(f"[eval] familia   : {model_meta.get('family')}  |  "
          f"{model_meta.get('params')}  |  {model_meta.get('size_gb')} GB")
    print(f"[eval] dataset   : {args.dataset} - {ds_cfg['display']}")
    print(f"[eval] modalidad : {ds_cfg.get('modality')}")
    print(f"[eval] instancias: {len(test_set)}")
    print(f"[eval] workers   : {args.workers}")
    if args.dry_run:
        print(f"[eval] modo      : DRY-RUN")

    # Evaluacion paralela
    results_map     = {}
    n_unclear       = 0
    checkpoint_lock = threading.Lock()
    partial_path    = runs_dir / f"partial_{safe_model}_{args.dataset}.jsonl"
    completed       = 0
    t0              = time.time()

    pbar = tqdm(total=len(test_set),
                desc=f"{args.model} x {args.dataset}", unit="meme")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                classify_meme, args.model,
                item["text"], item["image_path"]
            ): i
            for i, item in enumerate(test_set)
        }

        for future in as_completed(futures):
            i    = futures[future]
            pred = future.result()
            results_map[i] = pred
            completed += 1
            pbar.update(1)

            if not args.dry_run and completed % SAVE_EVERY == 0:
                with checkpoint_lock:
                    with partial_path.open("w", encoding="utf-8") as pf:
                        for idx in sorted(results_map.keys()):
                            row = {
                                "idx":        idx,
                                "text":       test_set[idx]["text"],
                                "image_path": test_set[idx]["image_path"],
                                "gold_label": test_set[idx]["label"],
                                "pred_label": results_map[idx],
                                "model":      args.model,
                                "dataset":    args.dataset,
                            }
                            pf.write(json.dumps(row, ensure_ascii=False) + "\n")
                tqdm.write(f"  [checkpoint] {completed}/{len(test_set)} guardado")

    pbar.close()
    elapsed = time.time() - t0

    # Reordena
    trues            = []
    preds            = []
    predictions_rows = []

    for i, item in enumerate(test_set):
        pred = results_map.get(i, "unclear")
        if pred == "unclear":
            n_unclear += 1
        trues.append(item["label"])
        preds.append(pred)
        predictions_rows.append({
            "idx":        i,
            "text":       item["text"],
            "image_path": item["image_path"],
            "gold_label": item["label"],
            "pred_label": pred,
            "model":      args.model,
            "dataset":    args.dataset,
        })

    # Dry-run
    if args.dry_run:
        print(f"\n[dry-run] predicciones : {preds}")
        print(f"[dry-run] gold         : {trues}")
        print(f"[dry-run] unclear      : {n_unclear}/{len(trues)}")
        return

    # Metricas
    labels_present = ["hate", "no_hate", "unclear"]
    macro_f1  = f1_score(trues, preds, average="macro",
                         labels=labels_present, zero_division=0)
    precision = precision_score(trues, preds, average="macro",
                                labels=labels_present, zero_division=0)
    recall    = recall_score(trues, preds, average="macro",
                             labels=labels_present, zero_division=0)
    accuracy  = accuracy_score(trues, preds)
    per_class = classification_report(
        trues, preds, labels=labels_present,
        output_dict=True, zero_division=0,
    )
    trues_bin    = [t for t, p in zip(trues, preds) if p != "unclear"]
    preds_bin    = [p for p in preds if p != "unclear"]
    macro_f1_bin = f1_score(trues_bin, preds_bin, average="macro",
                            zero_division=0) if trues_bin else 0

    print(f"\n[eval] Resultados")
    print(f"[eval] macro-F1 (3 clases) : {macro_f1:.4f}")
    print(f"[eval] macro-F1 (binario)  : {macro_f1_bin:.4f}")
    print(f"[eval] accuracy            : {accuracy:.4f}")
    print(f"[eval] hate F1             : {per_class.get('hate', {}).get('f1-score', 0):.4f}")
    print(f"[eval] unclear             : {n_unclear}/{len(test_set)}  ({n_unclear/len(test_set)*100:.1f}%)")
    print(f"[eval] tiempo              : {elapsed:.0f}s  ({elapsed/len(test_set):.1f}s/meme)")

    pred_fname = f"{ts.strftime('%Y-%m-%d')}_{safe_model}_{args.dataset}_predictions.jsonl"
    fname      = f"{ts.strftime('%Y-%m-%d')}_{safe_model}_{args.dataset}.json"

    result = {
        "run_id":          fname,
        "timestamp":       ts.isoformat(),
        "model":           args.model,
        "family":          model_meta.get("family", ""),
        "params":          model_meta.get("params", ""),
        "size_gb":         model_meta.get("size_gb"),
        "dataset":         args.dataset,
        "dataset_display": ds_cfg["display"],
        "modality":        ds_cfg.get("modality", "image+text"),
        "n_instances":     len(test_set),
        "n_unclear":       n_unclear,
        "n_workers":       args.workers,
        "metrics": {
            "macro_f1":        round(macro_f1,     4),
            "macro_f1_binary": round(macro_f1_bin, 4),
            "precision":       round(precision,    4),
            "recall":          round(recall,       4),
            "accuracy":        round(accuracy,     4),
        },
        "per_class":    per_class,
        "label_distribution": {
            "gold": {
                "hate":    trues.count("hate"),
                "no_hate": trues.count("no_hate"),
            },
            "pred": {
                "hate":    preds.count("hate"),
                "no_hate": preds.count("no_hate"),
                "unclear": preds.count("unclear"),
            },
        },
        "elapsed_seconds":  round(elapsed, 1),
        "prompt_hash":      "v1_multimodal",
        "predictions_file": f"results/predictions/{pred_fname}",
    }

    out_path = runs_dir / fname
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[eval] saved -> results/runs/{fname}")

    pred_path = pred_dir / pred_fname
    with pred_path.open("w", encoding="utf-8") as pf:
        for row in predictions_rows:
            pf.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[eval] saved -> results/predictions/{pred_fname}")

    if partial_path.exists():
        partial_path.unlink()


if __name__ == "__main__":
    main()
