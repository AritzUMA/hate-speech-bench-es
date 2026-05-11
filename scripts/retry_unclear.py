#!/usr/bin/env python3
"""
Reintenta clasificar los tweets con pred_label == 'unclear' en un run existente.
Actualiza el jsonl de predicciones y recalcula las metricas en results/runs/.

Uso:
    python3 scripts/retry_unclear.py --model llama3.1:8b --dataset encarni
    python3 scripts/retry_unclear.py --all                  # todos los runs con unclear
    python3 scripts/retry_unclear.py --threshold 0.95       # solo runs con cobertura < 95%
"""
import argparse
import json
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import ollama
import yaml
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score,
)
from tqdm import tqdm

REPO_ROOT  = Path(__file__).parent.parent
N_WORKERS  = 16

SYSTEM_PROMPT = """Eres un clasificador de tweets en espanol.
Analiza cada tweet y responde solo con un objeto JSON valido exactamente en este formato:
{
  "is_hate_speech": "Yes" | "No" | "Unclear",
  "rationale": "Breve explicacion (1-2 frases)"
}
No incluyas texto adicional, ni comentarios, ni encabezados ni markdown. SOLO JSON puro."""


def classify_tweet(model: str, text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": text},
    ]
    try:
        response = ollama.chat(
            model=model,
            messages=messages,
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
        if val == "no":
            return "no_hate"
        return "unclear"
    except Exception:
        return "unclear"


def get_runs_with_unclear(threshold: float) -> list[dict]:
    runs_dir = REPO_ROOT / "results" / "runs"
    result   = []
    for f in sorted(runs_dir.glob("*.json")):
        try:
            r = json.loads(f.read_text())
            n = r.get("n_instances", 0)
            u = r.get("n_unclear", 0)
            if n == 0 or u == 0:
                continue
            coverage = (n - u) / n
            if coverage < threshold:
                result.append({
                    "model":   r["model"],
                    "dataset": r["dataset"],
                    "n":       n,
                    "unclear": u,
                    "coverage": round(coverage * 100, 1),
                    "run_file": f,
                })
        except Exception:
            pass
    return result


def find_pred_file(model: str, dataset: str) -> Path | None:
    pred_dir   = REPO_ROOT / "results" / "predictions"
    safe_model = model.replace(":", "-").replace("/", "-")
    matches    = list(pred_dir.glob(f"*{safe_model}_{dataset}_predictions.jsonl"))
    if not matches:
        return None
    return sorted(matches)[-1]  # el mas reciente


def retry_run(model: str, dataset: str, workers: int = N_WORKERS) -> bool:
    pred_path = find_pred_file(model, dataset)
    if not pred_path:
        print(f"  [skip] No se encontro predictions file para {model} x {dataset}")
        return False

    # Lee predicciones existentes
    rows = []
    with pred_path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    unclear_idx = [i for i, r in enumerate(rows) if r["pred_label"] == "unclear"]
    if not unclear_idx:
        print(f"  [skip] {model} x {dataset} — no hay unclear")
        return False

    print(f"\n[retry] {model} x {dataset}")
    print(f"  unclear: {len(unclear_idx)}/{len(rows)}")

    # Reclasifica en paralelo
    results_map = {}
    lock        = threading.Lock()
    completed   = 0

    pbar = tqdm(total=len(unclear_idx), desc=f"retry {model[:25]} x {dataset}", unit="tweet")

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(classify_tweet, model, rows[i]["text"]): i
            for i in unclear_idx
        }
        for future in as_completed(futures):
            i    = futures[future]
            pred = future.result()
            results_map[i] = pred
            completed += 1
            pbar.update(1)

    pbar.close()

    # Actualiza predicciones
    n_recovered = 0
    for i, pred in results_map.items():
        if pred != "unclear":
            n_recovered += 1
        rows[i]["pred_label"] = pred
        rows[i]["correct"]    = pred == rows[i]["gold_label"]

    print(f"  recuperados: {n_recovered}/{len(unclear_idx)}")

    # Guarda predicciones actualizadas
    with pred_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Recalcula metricas
    trues = [r["gold_label"] for r in rows]
    preds = [r["pred_label"] for r in rows]

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
    n_unclear_new = preds.count("unclear")

    # Actualiza el run JSON
    runs_dir = REPO_ROOT / "results" / "runs"
    safe_model = model.replace(":", "-").replace("/", "-")
    run_files  = list(runs_dir.glob(f"*{safe_model}_{dataset}.json"))
    if not run_files:
        print(f"  [warn] No se encontro run file para {model} x {dataset}")
        return False

    run_path = sorted(run_files)[-1]
    run_data = json.loads(run_path.read_text())

    run_data["n_unclear"]       = n_unclear_new
    run_data["metrics"]["macro_f1"]        = round(macro_f1,     4)
    run_data["metrics"]["macro_f1_binary"] = round(macro_f1_bin, 4)
    run_data["metrics"]["precision"]       = round(precision,    4)
    run_data["metrics"]["recall"]          = round(recall,       4)
    run_data["metrics"]["accuracy"]        = round(accuracy,     4)
    run_data["per_class"]                  = per_class
    run_data["label_distribution"]["pred"] = {
        "hate":    preds.count("hate"),
        "no_hate": preds.count("no_hate"),
        "unclear": n_unclear_new,
    }
    run_data["retry_unclear"] = True

    run_path.write_text(json.dumps(run_data, indent=2, ensure_ascii=False))

    print(f"  macro-F1: {macro_f1:.4f}  hate-F1: {per_class.get('hate',{}).get('f1-score',0):.4f}  unclear restantes: {n_unclear_new}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default=None)
    parser.add_argument("--dataset",   default=None)
    parser.add_argument("--all",       action="store_true",
                        help="Reintenta todos los runs con unclear")
    parser.add_argument("--threshold", type=float, default=1.0,
                        help="Solo runs con cobertura < threshold (default: 1.0 = todos con algun unclear)")
    parser.add_argument("--workers",   type=int, default=N_WORKERS)
    args = parser.parse_args()

    if args.model and args.dataset:
        retry_run(args.model, args.dataset, args.workers)
        return

    if args.all or args.threshold < 1.0:
        runs = get_runs_with_unclear(args.threshold)
        if not runs:
            print("No hay runs con unclear por debajo del umbral.")
            return
        print(f"Runs a reintentar: {len(runs)}")
        for r in sorted(runs, key=lambda x: x["coverage"]):
            print(f"  {r['coverage']:>5.1f}%  {r['model']:<45} {r['dataset']}")
        print()
        for r in sorted(runs, key=lambda x: x["coverage"]):
            retry_run(r["model"], r["dataset"], args.workers)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
