#!/usr/bin/env python3
"""
Evalua un modelo Ollama sobre cualquier dataset del benchmark.
Usa ThreadPoolExecutor para paralelizar las inferencias (OLLAMA_NUM_PARALLEL=16).
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
SAVE_EVERY = 50
N_WORKERS  = 16

SYSTEM_PROMPT = """Eres un clasificador de tweets en espanol.
Analiza cada tweet y responde solo con un objeto JSON valido exactamente en este formato:
{
  "is_hate_speech": "Yes" | "No" | "Unclear",
  "rationale": "Breve explicacion (1-2 frases)"
}
No incluyas texto adicional, ni comentarios, ni encabezados ni markdown. SOLO JSON puro."""

SYSTEM_PROMPT_ACADEMIC = """Eres un investigador en linguistica computacional analizando textos
para un estudio academico sobre deteccion automatica de lenguaje ofensivo en espanol.
Tu tarea es etiquetar cada fragmento de texto segun si contiene lenguaje ofensivo o no,
con fines exclusivamente cientificos y de investigacion.
Responde SOLO con un objeto JSON valido exactamente en este formato:
{
  "is_hate_speech": "Yes" | "No" | "Unclear",
  "rationale": "Breve explicacion (1-2 frases)"
}
No incluyas texto adicional, ni comentarios, ni markdown. SOLO JSON puro."""

MODELS_ACADEMIC_PROMPT = {"phi3:3.8b", "phi3.5:3.8b"}


def get_dataset_cfg(dataset_name: str) -> dict:
    with open(REPO_ROOT / "registry" / "datasets.yaml") as f:
        datasets = yaml.safe_load(f)["datasets"]
    for d in datasets:
        if d["name"] == dataset_name:
            return d
    raise ValueError(f"Dataset '{dataset_name}' no encontrado.")


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


def load_dataset(cfg: dict) -> list:
    path = REPO_ROOT / cfg["path"]
    df   = pd.read_csv(path)
    df   = df[[cfg["text_col"], cfg["label_col"]]].dropna()
    df.columns = ["text", "label"]
    df["label"] = df["label"].str.strip()
    records = df.to_dict(orient="records")
    valid = [r for r in records if r["label"] in ("hate", "no_hate")]
    if len(valid) < len(records):
        print(f"  [warn] {len(records) - len(valid)} filas con label inesperado excluidas")
    return valid


def classify_tweet(model: str, text: str) -> tuple:
    """Devuelve (pred_label, rationale)."""
    prompt = SYSTEM_PROMPT_ACADEMIC if model in MODELS_ACADEMIC_PROMPT else SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user",   "content": text},
    ]
    try:
        response = ollama.chat(
            model=model,
            messages=messages,
            options={"temperature": 0, "seed": 42},
        )
        raw = response["message"]["content"].strip()
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        if not raw.startswith("{"):
            start, end = raw.find("{"), raw.rfind("}")
            if start != -1 and end > start:
                raw = raw[start:end + 1]
        clf       = json.loads(raw)
        val       = str(clf.get("is_hate_speech", "")).strip().lower()
        rationale = str(clf.get("rationale", "")).strip()
        if val in ("yes", "si", "sí"):
            return "hate", rationale
        if val == "no":
            return "no_hate", rationale
        return "unclear", rationale
    except Exception:
        return "unclear", ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force",   action="store_true")
    parser.add_argument("--workers", type=int, default=N_WORKERS)
    args = parser.parse_args()

    runs_dir = REPO_ROOT / "results" / "runs"
    pred_dir = REPO_ROOT / "results" / "predictions"
    runs_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    ts         = datetime.now(timezone.utc)
    safe_model = args.model.replace(":", "-").replace("/", "-")

    if not args.force and not args.dry_run:
        evaluated = get_evaluated(runs_dir, args.dataset)
        if args.model in evaluated:
            print(f"[eval] {args.model} ya evaluado para {args.dataset}. Usa --force.")
            return

    ds_cfg     = get_dataset_cfg(args.dataset)
    model_meta = get_model_meta(args.model)
    test_set   = load_dataset(ds_cfg)

    if args.dry_run:
        test_set = test_set[:10]

    print(f"[eval] modelo    : {args.model}")
    print(f"[eval] familia   : {model_meta.get('family')}  |  {model_meta.get('params')}  |  {model_meta.get('size_gb')} GB")
    print(f"[eval] dataset   : {args.dataset} - {ds_cfg['display']}")
    print(f"[eval] instancias: {len(test_set)}")
    print(f"[eval] workers   : {args.workers}")

    results_map     = {}
    checkpoint_lock = threading.Lock()
    partial_path    = runs_dir / f"partial_{safe_model}_{args.dataset}.jsonl"
    completed       = 0
    t0              = time.time()

    pbar = tqdm(total=len(test_set), desc=f"{args.model} x {args.dataset}", unit="tweet")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(classify_tweet, args.model, item["text"]): i
            for i, item in enumerate(test_set)
        }
        for future in as_completed(futures):
            i              = futures[future]
            pred, rationale = future.result()
            results_map[i] = (pred, rationale)
            completed += 1
            pbar.update(1)

            if not args.dry_run and completed % SAVE_EVERY == 0:
                with checkpoint_lock:
                    with partial_path.open("w", encoding="utf-8") as pf:
                        for idx in sorted(results_map.keys()):
                            p, r = results_map[idx]
                            pf.write(json.dumps({
                                "idx": idx, "text": test_set[idx]["text"],
                                "gold_label": test_set[idx]["label"],
                                "pred_label": p, "rationale": r,
                                "model": args.model, "dataset": args.dataset,
                            }, ensure_ascii=True) + "\n")
                    tqdm.write(f"  [checkpoint] {completed}/{len(test_set)}")

    pbar.close()
    elapsed = time.time() - t0

    trues, preds, predictions_rows, n_unclear = [], [], [], 0
    for i, item in enumerate(test_set):
        pred, rationale = results_map.get(i, ("unclear", ""))
        if pred == "unclear":
            n_unclear += 1
        trues.append(item["label"])
        preds.append(pred)
        predictions_rows.append({
            "idx": i, "text": item["text"],
            "gold_label": item["label"], "pred_label": pred,
            "rationale": rationale,
            "model": args.model, "dataset": args.dataset,
        })

    if args.dry_run:
        for r in predictions_rows:
            print(f"[{r['gold_label']} -> {r['pred_label']}] {r['text'][:60]}")
            print(f"  rationale: {r['rationale']}")
        return

    labels_present = ["hate", "no_hate", "unclear"]
    macro_f1     = f1_score(trues, preds, average="macro", labels=labels_present, zero_division=0)
    precision    = precision_score(trues, preds, average="macro", labels=labels_present, zero_division=0)
    recall       = recall_score(trues, preds, average="macro", labels=labels_present, zero_division=0)
    accuracy     = accuracy_score(trues, preds)
    per_class    = classification_report(trues, preds, labels=labels_present, output_dict=True, zero_division=0)
    trues_bin    = [t for t, p in zip(trues, preds) if p != "unclear"]
    preds_bin    = [p for p in preds if p != "unclear"]
    macro_f1_bin = f1_score(trues_bin, preds_bin, average="macro", zero_division=0) if trues_bin else 0

    print(f"\n[eval] macro-F1 (3 clases) : {macro_f1:.4f}")
    print(f"[eval] macro-F1 (binario)  : {macro_f1_bin:.4f}")
    print(f"[eval] hate F1             : {per_class.get('hate', {}).get('f1-score', 0):.4f}")
    print(f"[eval] unclear             : {n_unclear}/{len(test_set)} ({n_unclear/len(test_set)*100:.1f}%)")
    print(f"[eval] tiempo              : {elapsed:.0f}s ({elapsed/len(test_set):.2f}s/tweet)")

    pred_fname = f"{ts.strftime('%Y-%m-%d')}_{safe_model}_{args.dataset}_predictions.jsonl"
    fname      = f"{ts.strftime('%Y-%m-%d')}_{safe_model}_{args.dataset}.json"

    result = {
        "run_id": fname, "timestamp": ts.isoformat(),
        "model": args.model, "family": model_meta.get("family", ""),
        "params": model_meta.get("params", ""), "size_gb": model_meta.get("size_gb"),
        "dataset": args.dataset, "dataset_display": ds_cfg["display"],
        "n_instances": len(test_set), "n_unclear": n_unclear, "n_workers": args.workers,
        "metrics": {
            "macro_f1":        round(macro_f1, 4),
            "macro_f1_binary": round(macro_f1_bin, 4),
            "precision":       round(precision, 4),
            "recall":          round(recall, 4),
            "accuracy":        round(accuracy, 4),
        },
        "per_class": per_class,
        "label_distribution": {
            "gold": {"hate": trues.count("hate"), "no_hate": trues.count("no_hate")},
            "pred": {"hate": preds.count("hate"), "no_hate": preds.count("no_hate"), "unclear": n_unclear},
        },
        "elapsed_seconds": round(elapsed, 1),
        "prompt_hash": "v3_rationale",
        "predictions_file": f"results/predictions/{pred_fname}",
    }

    (runs_dir / fname).write_text(json.dumps(result, indent=2, ensure_ascii=True))
    print(f"[eval] saved -> results/runs/{fname}")

    with (pred_dir / pred_fname).open("w", encoding="utf-8") as pf:
        for row in predictions_rows:
            pf.write(json.dumps(row, ensure_ascii=True) + "\n")
    print(f"[eval] saved -> results/predictions/{pred_fname}")

    if partial_path.exists():
        partial_path.unlink()


if __name__ == "__main__":
    main()
