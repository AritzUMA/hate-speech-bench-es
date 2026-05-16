#!/usr/bin/env python3
"""
Para cada run JSON, detecta observaciones sin prediccion en el JSONL
y las clasifica. Actualiza el JSONL y recalcula las metricas.

Uso:
    python3 scripts/complete_missing.py
    python3 scripts/complete_missing.py --model llama3.1:8b
    python3 scripts/complete_missing.py --model llama3.1:8b --dataset encarni
"""
import argparse
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

REPO_ROOT  = Path(__file__).parent.parent
RUNS_DIR   = REPO_ROOT / "results" / "runs"
PRED_DIR   = REPO_ROOT / "results" / "predictions"
N_WORKERS  = 16
SAVE_EVERY = 50

SYSTEM_PROMPT = """Eres un clasificador de tweets en espanol.
Analiza cada tweet y responde solo con un objeto JSON valido exactamente en este formato:
{
  "is_hate_speech": "Yes" | "No" | "Unclear",
  "rationale": "Breve explicacion (1-2 frases)"
}
No incluyas texto adicional, ni comentarios, ni encabezados ni markdown. SOLO JSON puro."""


def get_dataset_cfg(dataset_name):
    with open(REPO_ROOT / "registry" / "datasets.yaml") as f:
        for d in yaml.safe_load(f)["datasets"]:
            if d["name"] == dataset_name:
                return d
    raise ValueError(f"Dataset {dataset_name} no encontrado")


def load_dataset(cfg):
    path = REPO_ROOT / cfg["path"]
    df   = pd.read_csv(path)[[cfg["text_col"], cfg["label_col"]]].dropna()
    df.columns = ["text", "label"]
    df["label"] = df["label"].str.strip()
    return df[df["label"].isin(["hate","no_hate"])].to_dict(orient="records")


def find_pred_file(model, dataset):
    safe = model.replace(":", "-").replace("/", "-")
    matches = sorted(PRED_DIR.glob(f"*_{safe}_{dataset}_predictions.jsonl"))
    return matches[-1] if matches else None


def classify_tweet(model, text):
    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
            options={"temperature": 0},
        )
        raw = response["message"]["content"].strip()
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        raw = raw.replace("```json","").replace("```","").strip()
        if not raw.startswith("{"):
            s, e = raw.find("{"), raw.rfind("}")
            if s != -1 and e > s:
                raw = raw[s:e+1]
        clf = json.loads(raw)
        val = str(clf.get("is_hate_speech","")).strip().lower()
        if val in ("yes","si","sí"): return "hate"
        if val == "no":              return "no_hate"
        return "unclear"
    except Exception:
        return "unclear"


def recalculate_metrics(run_path, pred_file, test_set):
    rows = {}
    with open(pred_file) as f:
        for line in f:
            r = json.loads(line)
            rows[r["idx"]] = r

    trues, preds = [], []
    for i, item in enumerate(test_set):
        pred = rows.get(i, {}).get("pred_label", "unclear")
        trues.append(item["label"])
        preds.append(pred)

    labels_present = ["hate","no_hate","unclear"]
    macro_f1  = f1_score(trues, preds, average="macro", labels=labels_present, zero_division=0)
    precision = precision_score(trues, preds, average="macro", labels=labels_present, zero_division=0)
    recall    = recall_score(trues, preds, average="macro", labels=labels_present, zero_division=0)
    accuracy  = accuracy_score(trues, preds)
    per_class = classification_report(trues, preds, labels=labels_present, output_dict=True, zero_division=0)
    trues_bin = [t for t,p in zip(trues,preds) if p!="unclear"]
    preds_bin = [p for p in preds if p!="unclear"]
    macro_f1_bin = f1_score(trues_bin, preds_bin, average="macro", zero_division=0) if trues_bin else 0
    n_unclear = preds.count("unclear")

    run = json.loads(run_path.read_text())
    run["metrics"] = {
        "macro_f1":        round(macro_f1,     4),
        "macro_f1_binary": round(macro_f1_bin, 4),
        "precision":       round(precision,    4),
        "recall":          round(recall,       4),
        "accuracy":        round(accuracy,     4),
    }
    run["per_class"]  = per_class
    run["n_unclear"]  = n_unclear
    run["n_instances"] = len(test_set)
    run["label_distribution"] = {
        "gold": {"hate": trues.count("hate"), "no_hate": trues.count("no_hate")},
        "pred": {"hate": preds.count("hate"), "no_hate": preds.count("no_hate"), "unclear": n_unclear},
    }
    run["completed_at"] = datetime.now(timezone.utc).isoformat()
    run_path.write_text(json.dumps(run, indent=2, ensure_ascii=False))
    return macro_f1, n_unclear


def process_run(run_path, filter_model=None, filter_dataset=None, dry_run=False):
    run = json.loads(run_path.read_text())
    model   = run["model"]
    dataset = run["dataset"]

    if filter_model   and model   != filter_model:   return
    if filter_dataset and dataset != filter_dataset: return
    if run.get("modality") == "image+text":          return

    pred_file = find_pred_file(model, dataset)
    if not pred_file:
        print(f"[skip] No hay predictions para {model} x {dataset}")
        return

    try:
        ds_cfg   = get_dataset_cfg(dataset)
        test_set = load_dataset(ds_cfg)
    except Exception as e:
        print(f"[skip] Error cargando dataset {dataset}: {e}")
        return

    # Leer predicciones existentes
    existing = {}
    with open(pred_file) as f:
        for line in f:
            try:
                r = json.loads(line)
                existing[r["idx"]] = r
            except Exception:
                pass

    # Detectar indices que faltan (no estan en el JSONL)
    missing_idxs = [i for i in range(len(test_set)) if i not in existing]

    if not missing_idxs:
        print(f"[ok] {model} x {dataset} — completado ({len(existing)}/{len(test_set)})")
        return

    coverage = len(existing) / len(test_set) * 100
    print(f"\n[complete] {model} x {dataset}")
    print(f"  existentes: {len(existing)}/{len(test_set)} ({coverage:.1f}%)")
    print(f"  a clasificar: {len(missing_idxs)}")

    # Clasificar los que faltan
    new_preds  = {}
    lock       = threading.Lock()
    completed  = 0
    t0         = time.time()

    pbar = tqdm(total=len(missing_idxs), desc=f"  {model[:30]} x {dataset}", unit="tweet")

    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {
            executor.submit(classify_tweet, model, test_set[i]["text"]): i
            for i in missing_idxs
        }
        for future in as_completed(futures):
            i    = futures[future]
            pred = future.result()
            new_preds[i] = pred
            completed += 1
            pbar.update(1)

            if completed % SAVE_EVERY == 0:
                with lock:
                    # Append nuevas predicciones al JSONL
                    with open(pred_file, "a", encoding="utf-8") as pf:
                        for idx, p in list(new_preds.items()):
                            if idx not in existing:
                                row = {
                                    "idx":        idx,
                                    "text":       test_set[idx]["text"],
                                    "gold_label": test_set[idx]["label"],
                                    "pred_label": p,
                                    "model":      model,
                                    "dataset":    dataset,
                                }
                                pf.write(json.dumps(row, ensure_ascii=False) + "\n")
                                existing[idx] = row
                    tqdm.write(f"  [checkpoint] {completed}/{len(missing_idxs)}")

    pbar.close()

    # Guardar predicciones restantes
    with open(pred_file, "a", encoding="utf-8") as pf:
        for idx, p in new_preds.items():
            if idx not in existing:
                row = {
                    "idx":        idx,
                    "text":       test_set[idx]["text"],
                    "gold_label": test_set[idx]["label"],
                    "pred_label": p,
                    "model":      model,
                    "dataset":    dataset,
                }
                pf.write(json.dumps(row, ensure_ascii=False) + "\n")
                existing[idx] = row

    # Recalcular metricas
    macro_f1, n_unclear = recalculate_metrics(run_path, pred_file, test_set)
    elapsed = time.time() - t0
    total   = len(test_set)
    print(f"  [done] macro_f1={macro_f1:.4f}  unclear={n_unclear}/{total}  tiempo={elapsed:.0f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=None)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="Solo muestra cuantas observaciones faltan sin clasificar")
    args = parser.parse_args()

    run_files = sorted(RUNS_DIR.glob("*.json"))
    run_files = [f for f in run_files if not f.name.startswith("partial_")]

    print(f"[complete_missing] {len(run_files)} runs a revisar")
    print(f"  workers: {N_WORKERS}\n")

    for run_path in run_files:
        process_run(run_path, args.model, args.dataset, args.dry_run)

    print("\n[complete_missing] Listo.")


if __name__ == "__main__":
    main()
