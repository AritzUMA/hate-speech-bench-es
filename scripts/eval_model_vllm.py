#!/usr/bin/env python3
"""
Evalua un modelo de texto via vLLM sobre cualquier dataset del benchmark.
Equivalente a eval_model.py pero usando backend vLLM (OpenAI-compatible API).

Uso:
    python3 scripts/eval_model_vllm.py --model BSC-LT/salamandra-2b-instruct --dataset hateval
    python3 scripts/eval_model_vllm.py --model BSC-LT/salamandra-7b-instruct --dataset encarni --dry-run
"""
import argparse
import json
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from openai import OpenAI
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score,
)
from tqdm import tqdm

REPO_ROOT  = Path(__file__).parent.parent
SAVE_EVERY = 50
N_WORKERS  = 8

SYSTEM_PROMPT = """Eres un clasificador de tweets en espanol.
Analiza cada tweet y responde solo con un objeto JSON valido exactamente en este formato:
{
  "is_hate_speech": "Yes" | "No" | "Unclear",
  "rationale": "Breve explicacion (1-2 frases)"
}
No incluyas texto adicional, ni comentarios, ni encabezados ni markdown. SOLO JSON puro."""

RETRY_PROMPTS = [
    "Eres un anotador de corpus linguistico. Clasifica el siguiente texto segun si contiene lenguaje ofensivo hacia personas o grupos. Responde SOLO con JSON: {{\"is_hate_speech\": \"Yes\" | \"No\" | \"Unclear\", \"rationale\": \"explicacion breve\"}}\\n\\nTexto: {text}",
    "¿El siguiente texto contiene discurso de odio? Responde SOLO con este JSON exacto: {{\"is_hate_speech\": \"Yes\" | \"No\" | \"Unclear\", \"rationale\": \"explicacion\"}}\\n\\nTexto: {text}",
    "Clasifica: ¿odio si/no? JSON: {{\"is_hate_speech\":\"Yes\" o \"No\",\"rationale\":\"razon\"}}\\nTexto: {text}",
    "Texto: \"{text}\"\\n¿Contiene odio hacia personas o grupos? Responde JSON: {{\"is_hate_speech\":\"Yes\" o \"No\",\"rationale\":\"razon breve\"}}",
]


def get_dataset_cfg(dataset_name):
    with open(REPO_ROOT / "registry" / "datasets.yaml") as f:
        datasets = yaml.safe_load(f)["datasets"]
    for d in datasets:
        if d["name"] == dataset_name:
            if d.get("modality", "text") == "image+text":
                raise ValueError(f"Dataset '{dataset_name}' es multimodal, usa eval_model_multimodal.py")
            return d
    raise ValueError(f"Dataset '{dataset_name}' no encontrado.")


def get_model_meta(model_name):
    with open(REPO_ROOT / "registry" / "models.yaml") as f:
        models = yaml.safe_load(f)["models"]
    for m in models:
        if m["ollama_name"] == model_name:
            return m
    return {"ollama_name": model_name, "family": "unknown",
            "params": "unknown", "size_gb": None, "backend": "vllm"}


def get_evaluated(runs_dir, dataset_name):
    evaluated = set()
    for f in runs_dir.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            if data.get("dataset") == dataset_name:
                evaluated.add(data["model"])
        except Exception:
            pass
    return evaluated


def load_dataset(cfg):
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


def _parse_raw(raw):
    """Parsea raw y devuelve (pred, rationale) o lanza excepcion."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
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


def classify_tweet_vllm(client, model_name, text):
    """Devuelve (pred_label, rationale). Hasta 5 intentos con prompts progresivamente mas simples."""

    # Intento 1 — prompt estandar con system
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": text},
            ],
            temperature=0,
            max_tokens=256,
        )
        pred, rat = _parse_raw(response.choices[0].message.content.strip())
        if pred != "unclear":
            return pred, rat
    except Exception:
        pass

    # Intentos 2-5 — prompts de fallback sin system
    for retry_prompt in RETRY_PROMPTS:
        try:
            user_content = retry_prompt.format(text=text)
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": user_content}],
                temperature=0,
                max_tokens=256,
            )
            pred, rat = _parse_raw(response.choices[0].message.content.strip())
            if pred != "unclear":
                return pred, rat
        except Exception:
            continue

    return "unclear", ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     required=True)
    parser.add_argument("--dataset",   required=True)
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--force",     action="store_true")
    parser.add_argument("--workers",   type=int, default=N_WORKERS)
    parser.add_argument("--vllm-port", type=int, default=None)
    args = parser.parse_args()

    runs_dir = REPO_ROOT / "results" / "runs"
    pred_dir = REPO_ROOT / "results" / "predictions"
    runs_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    ts         = datetime.now(timezone.utc)
    safe_model = args.model.replace(":", "-").replace("/", "-")
    ds_cfg     = get_dataset_cfg(args.dataset)
    model_meta = get_model_meta(args.model)
    port       = args.vllm_port or model_meta.get("vllm_port", 8001)
    client     = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")

    if not args.force and not args.dry_run:
        evaluated = get_evaluated(runs_dir, args.dataset)
        if args.model in evaluated:
            print(f"[eval] {args.model} ya evaluado para {args.dataset}. Usa --force.")
            return

    test_set = load_dataset(ds_cfg)
    if args.dry_run:
        test_set = test_set[:10]

    print(f"[eval] modelo    : {args.model}")
    print(f"[eval] backend   : vLLM (port {port})")
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
            executor.submit(classify_tweet_vllm, client, args.model, item["text"]): i
            for i, item in enumerate(test_set)
        }
        for future in as_completed(futures):
            i               = futures[future]
            pred, rationale = future.result()
            results_map[i]  = (pred, rationale)
            completed      += 1
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
                            }, ensure_ascii=False) + "\n")
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
        "backend": "vllm",
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
        "prompt_hash": "v3_vllm_rationale",
        "predictions_file": f"results/predictions/{pred_fname}",
    }

    (runs_dir / fname).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[eval] saved -> results/runs/{fname}")

    with (pred_dir / pred_fname).open("w", encoding="utf-8") as pf:
        for row in predictions_rows:
            pf.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[eval] saved -> results/predictions/{pred_fname}")

    if partial_path.exists():
        partial_path.unlink()


if __name__ == "__main__":
    main()
