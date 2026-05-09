#!/usr/bin/env python3
"""
Evalua un modelo Ollama sobre cualquier dataset del benchmark.
Usa ThreadPoolExecutor para paralelizar las inferencias (OLLAMA_NUM_PARALLEL=4).

Uso:
    python3 scripts/eval_model.py --model llama3.1:8b --dataset hateval
    python3 scripts/eval_model.py --model llama3.1:8b --dataset encarni --dry-run
    python3 scripts/eval_model.py --model gemma3:12b --dataset misocorpus --force
"""
import argparse
import json
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
SAVE_EVERY = 50
N_WORKERS  = 8

# Prompt
SYSTEM_PROMPT = """Eres un clasificador de tweets en espanol.
Analiza cada tweet y responde solo con un objeto JSON valido exactamente en este formato:
{
  "is_hate_speech": "Yes" | "No" | "Unclear",
  "rationale": "Breve explicacion (1-2 frases)"
}
No incluyas texto adicional, ni comentarios, ni encabezados ni markdown. SOLO JSON puro."""


# Registry

def get_dataset_cfg(dataset_name: str) -> dict:
    with open(REPO_ROOT / "registry" / "datasets.yaml") as f:
        datasets = yaml.safe_load(f)["datasets"]
    for d in datasets:
        if d["name"] == dataset_name:
            return d
    raise ValueError(f"Dataset '{dataset_name}' no encontrado en datasets.yaml.\n"
                     f"Disponibles: {[d['name'] for d in datasets]}")


def get_model_meta(model_name: str) -> dict:
    with open(REPO_ROOT / "registry" / "models.yaml") as f:
        models = yaml.safe_load(f)["models"]
    for m in models:
        if m["ollama_name"] == model_name:
            return m
    return {"ollama_name": model_name, "family": "unknown",
            "params": "unknown", "size_gb": None}


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
    df   = df[[cfg["text_col"], cfg["label_col"]]].dropna()
    df.columns = ["text", "label"]
    df["label"] = df["label"].str.strip()
    records = df.to_dict(orient="records")
    valid = [r for r in records if r["label"] in ("hate", "no_hate")]
    if len(valid) < len(records):
        print(f"  [warn] {len(records) - len(valid)} filas con label inesperado - excluidas")
    return valid


# Clasificador

def classify_tweet(model: str, text: str, max_retries: int = 3):
    """
    Llama a Ollama y devuelve 'hate' / 'no_hate' / None.
    Tras max_retries fallidos devuelve None -> el caller usa fallback no_hate.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": text},
    ]

    for attempt in range(1, max_retries + 1):
        try:
            response = ollama.chat(
                model=model,
                messages=messages,
                options={"temperature": 0, "seed": 42},
            )
            raw = response["message"]["content"].strip()

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
                return "no_hate"

            tqdm.write(f"  [retry {attempt}/{max_retries}] valor inesperado: '{val}'")

        except json.JSONDecodeError as e:
            tqdm.write(f"  [retry {attempt}/{max_retries}] JSONDecodeError: {e}")
        except Exception as e:
            tqdm.write(f"  [retry {attempt}/{max_retries}] {type(e).__name__}: {e}")

        time.sleep(2)

    return None


# Main

def main():
    parser = argparse.ArgumentParser(
        description="Evalua un modelo Ollama sobre un dataset del benchmark")
    parser.add_argument("--model",   required=True,
                        help="Nombre Ollama, ej: llama3.1:8b")
    parser.add_argument("--dataset", required=True,
                        help="Nombre del dataset, ej: hateval, encarni, chileno...")
    parser.add_argument("--dry-run", action="store_true",
                        help="Clasifica solo los primeros 10 tweets sin guardar")
    parser.add_argument("--force",   action="store_true",
                        help="Evalua aunque ya exista resultado en results/runs/")
    parser.add_argument("--workers", type=int, default=N_WORKERS,
                        help=f"Peticiones paralelas a Ollama (default: {N_WORKERS})")
    args = parser.parse_args()

    # Directorios
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
        test_set = test_set[:10]

    print(f"[eval] modelo    : {args.model}")
    print(f"[eval] familia   : {model_meta.get('family')}  |  "
          f"{model_meta.get('params')}  |  {model_meta.get('size_gb')} GB")
    print(f"[eval] dataset   : {args.dataset} - {ds_cfg['display']}")
    print(f"[eval] instancias: {len(test_set)}")
    print(f"[eval] workers   : {args.workers}")
    if args.dry_run:
        print(f"[eval] modo      : DRY-RUN (solo {len(test_set)} tweets, sin guardar)")

    # Evaluacion paralela
    results_map     = {}
    n_sin_pred      = 0
    checkpoint_lock = threading.Lock()
    partial_path    = runs_dir / f"partial_{safe_model}_{args.dataset}.jsonl"
    completed       = 0
    t0              = time.time()

    pbar = tqdm(total=len(test_set),
                desc=f"{args.model} x {args.dataset}", unit="tweet")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(classify_tweet, args.model, item["text"]): i
            for i, item in enumerate(test_set)
        }

        for future in as_completed(futures):
            i    = futures[future]
            pred = future.result()
            results_map[i] = pred
            completed += 1
            pbar.update(1)

            # Checkpoint parcial cada SAVE_EVERY tweets
            if not args.dry_run and completed % SAVE_EVERY == 0:
                with checkpoint_lock:
                    with partial_path.open("w", encoding="utf-8") as pf:
                        for idx in sorted(results_map.keys()):
                            row = {
                                "idx":        idx,
                                "text":       test_set[idx]["text"],
                                "gold_label": test_set[idx]["label"],
                                "pred_label": results_map[idx] or "no_hate",
                                "correct":    (results_map[idx] or "no_hate") == test_set[idx]["label"],
                                "model":      args.model,
                                "dataset":    args.dataset,
                            }
                            pf.write(json.dumps(row, ensure_ascii=False) + "\n")
                    tqdm.write(f"  [checkpoint] {completed}/{len(test_set)} guardado")

    pbar.close()
    elapsed = time.time() - t0

    # Reordena resultados por idx
    trues            = []
    preds            = []
    predictions_rows = []

    for i, item in enumerate(test_set):
        pred = results_map.get(i)
        if pred is None:
            n_sin_pred += 1
            pred = "no_hate"
        trues.append(item["label"])
        preds.append(pred)
        predictions_rows.append({
            "idx":        i,
            "text":       item["text"],
            "gold_label": item["label"],
            "pred_label": pred,
            "correct":    pred == item["label"],
            "model":      args.model,
            "dataset":    args.dataset,
        })

    # Dry-run: solo mostrar
    if args.dry_run:
        print(f"\n[dry-run] predicciones : {preds}")
        print(f"[dry-run] gold         : {[r['label'] for r in test_set]}")
        correct = sum(p == g for p, g in zip(preds, trues))
        print(f"[dry-run] correctas    : {correct}/{len(trues)}")
        return

    # Metricas
    macro_f1  = f1_score(trues, preds, average="macro",    zero_division=0)
    precision = precision_score(trues, preds, average="macro", zero_division=0)
    recall    = recall_score(trues, preds, average="macro",    zero_division=0)
    accuracy  = accuracy_score(trues, preds)
    per_class = classification_report(
        trues, preds,
        labels=["hate", "no_hate"],
        output_dict=True,
        zero_division=0,
    )

    print(f"\n[eval] Resultados")
    print(f"[eval] macro-F1   : {macro_f1:.4f}")
    print(f"[eval] precision  : {precision:.4f}")
    print(f"[eval] recall     : {recall:.4f}")
    print(f"[eval] accuracy   : {accuracy:.4f}")
    print(f"[eval] hate F1    : {per_class.get('hate', {}).get('f1-score', 0):.4f}")
    print(f"[eval] no_hate F1 : {per_class.get('no_hate', {}).get('f1-score', 0):.4f}")
    print(f"[eval] sin pred   : {n_sin_pred}")
    print(f"[eval] tiempo     : {elapsed:.0f}s  ({elapsed/len(test_set):.2f}s/tweet)")

    # Guardar metricas
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
        "n_instances":     len(test_set),
        "n_sin_pred":      n_sin_pred,
        "n_workers":       args.workers,
        "metrics": {
            "macro_f1":  round(macro_f1,  4),
            "precision": round(precision, 4),
            "recall":    round(recall,    4),
            "accuracy":  round(accuracy,  4),
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
            },
        },
        "elapsed_seconds":  round(elapsed, 1),
        "prompt_hash":      "v1",
        "predictions_file": f"results/predictions/{pred_fname}",
    }

    out_path = runs_dir / fname
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[eval] saved -> results/runs/{fname}")

    # Guardar predicciones por tweet
    pred_path = pred_dir / pred_fname
    with pred_path.open("w", encoding="utf-8") as pf:
        for row in predictions_rows:
            pf.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[eval] saved -> results/predictions/{pred_fname}")

    # Borra el partial si existia
    if partial_path.exists():
        partial_path.unlink()


if __name__ == "__main__":
    main()
