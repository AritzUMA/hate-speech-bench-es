#!/usr/bin/env python3
"""
Evalua un modelo vision sobre cualquier dataset multimodal del benchmark.
Soporta dos backends:
  - ollama: modelos locales via Ollama (default)
  - vllm:   modelos HuggingFace servidos via vLLM (OpenAI-compatible API)
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

import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score, classification_report,
    f1_score, precision_score, recall_score,
)
from tqdm import tqdm

REPO_ROOT        = Path(__file__).parent.parent
SAVE_EVERY       = 20
N_WORKERS_OLLAMA = 1
N_WORKERS_VLLM   = 8
RUNS_DIR_MM      = REPO_ROOT / "results" / "runs_multimodal"
PRED_DIR_MM      = REPO_ROOT / "results" / "predictions_multimodal"

SYSTEM_PROMPT = """Eres un clasificador de memes en espanol.
Analiza la imagen y el texto del meme y determina si contiene discurso de odio.
Responde SOLO con un objeto JSON exactamente en este formato:
{
  "is_hate_speech": "Yes" | "No" | "Unclear",
  "rationale": "Breve explicacion (1-2 frases)"
}
No incluyas texto adicional, ni comentarios, ni markdown. SOLO JSON puro."""


def get_dataset_cfg(dataset_name):
    with open(REPO_ROOT / "registry" / "datasets.yaml") as f:
        datasets = yaml.safe_load(f)["datasets"]
    for d in datasets:
        if d["name"] == dataset_name:
            if d.get("modality") != "image+text":
                raise ValueError(f"Dataset '{dataset_name}' no es multimodal.")
            return d
    raise ValueError(f"Dataset '{dataset_name}' no encontrado.")


def get_model_meta(model_name):
    with open(REPO_ROOT / "registry" / "models.yaml") as f:
        models = yaml.safe_load(f)["models"]
    for m in models:
        if m["ollama_name"] == model_name:
            return m
    return {"ollama_name": model_name, "family": "unknown",
            "params": "unknown", "size_gb": None,
            "modality": "image+text", "backend": "ollama"}


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
    df   = df[[cfg["text_col"], cfg["image_col"], cfg["label_col"]]].dropna()
    df.columns = ["text", "image_path", "label"]
    df["label"] = df["label"].str.strip()
    records = df.to_dict(orient="records")
    return [r for r in records if r["label"] in ("hate", "no_hate")]


def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def parse_response(raw):
    """Devuelve (pred_label, rationale). no_hate antes que hate."""
    if not raw:
        return "unclear", ""

    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    low = raw.lower()

    # 1. JSON estricto
    try:
        blob = raw
        if not blob.startswith("{"):
            s, e = blob.find("{"), blob.rfind("}")
            if s != -1 and e > s:
                blob = blob[s:e+1]
        clf       = json.loads(blob)
        val       = str(clf.get("is_hate_speech", clf.get("label", ""))).strip().lower()
        val       = val.replace("-", "_").replace(" ", "_")
        rationale = str(clf.get("rationale", clf.get("explanation", ""))).strip()
        if val in ("yes", "si", "hate"):
            return "hate", rationale
        if val in ("no", "no_hate", "nohate"):
            return "no_hate", rationale
        if val == "unclear":
            return "unclear", rationale
    except Exception:
        pass

    # 2. Texto libre — no_hate primero
    if re.search(r"no[_\s-]?hate", low):
        return "no_hate", ""
    if re.search(r"no\s+(es|contiene)\s+(discurso\s+de\s+)?odio", low):
        return "no_hate", ""
    if re.search(r"no\s+es\s+(hate|odio|discurso)", low):
        return "no_hate", ""
    if re.search(r"\bhate\b", low):
        return "hate", ""
    if re.search(r"(es|contiene)\s+(discurso\s+de\s+)?odio", low):
        return "hate", ""
    if re.search(r"\byes\b", low):
        return "hate", ""
    if re.search(r"^\s*no\b", low):
        return "no_hate", ""

    return "unclear", ""


def classify_meme_ollama(model, text, image_path):
    import ollama
    try:
        img_b64     = image_to_base64(image_path)
        caption_fmt = text.replace("<sep>", "\n").strip()
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f'Texto del meme:\n"{caption_fmt}"\n\nClasifica este meme.', "images": [img_b64]},
            ],
            options={"temperature": 0},
        )
        raw = response["message"]["content"].strip()
        pred, rationale = parse_response(raw)
        return pred, rationale, raw
    except Exception:
        return "unclear", "", ""


# Modelos que no soportan system role
MODELS_NO_SYSTEM = {"OpenGVLab/InternVL2-8B", "OpenGVLab/InternVL2-4B"}

def classify_meme_vllm(client, model_name, text, image_path):
    try:
        img_b64     = image_to_base64(image_path)
        caption_fmt = text.replace("<sep>", "\n").strip()

        # InternVL2 y similares no soportan system role — se fusiona con user
        if model_name in MODELS_NO_SYSTEM:
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": f'{SYSTEM_PROMPT}\n\nTexto del meme:\n"{caption_fmt}"\n\nClasifica este meme.'},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ]},
            ]
        else:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "text", "text": f'Texto del meme:\n"{caption_fmt}"\n\nClasifica este meme.'},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                ]},
            ]

        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_tokens=256,
        )
        raw = response.choices[0].message.content.strip()
        pred, rationale = parse_response(raw)
        return pred, rationale, raw
    except Exception:
        return "unclear", "", ""


def save_checkpoint(partial_path, results_map, test_set, model, dataset):
    with partial_path.open("w", encoding="utf-8") as pf:
        for idx in sorted(results_map.keys()):
            p, r, rw = results_map[idx]
            pf.write(json.dumps({
                "idx": idx,
                "text": test_set[idx]["text"],
                "image_path": test_set[idx]["image_path"],
                "gold_label": test_set[idx]["label"],
                "pred_label": p,
                "rationale": r,
                "raw_response": rw,
                "model": model,
                "dataset": dataset,
            }, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     required=True)
    parser.add_argument("--dataset",   required=True)
    parser.add_argument("--dry-run",   action="store_true")
    parser.add_argument("--force",     action="store_true")
    parser.add_argument("--workers",   type=int, default=None)
    parser.add_argument("--vllm-port", type=int, default=None)
    args = parser.parse_args()

    RUNS_DIR_MM.mkdir(parents=True, exist_ok=True)
    PRED_DIR_MM.mkdir(parents=True, exist_ok=True)

    ts         = datetime.now(timezone.utc)
    safe_model = args.model.replace(":", "-").replace("/", "-")
    ds_cfg     = get_dataset_cfg(args.dataset)
    model_meta = get_model_meta(args.model)
    backend    = model_meta.get("backend", "ollama")

    if backend == "vllm":
        from openai import OpenAI
        port      = args.vllm_port or model_meta.get("vllm_port", 8001)
        client    = OpenAI(base_url=f"http://localhost:{port}/v1", api_key="dummy")
        n_workers = args.workers or N_WORKERS_VLLM
        classify_fn = lambda text, img: classify_meme_vllm(client, args.model, text, img)
        print(f"[eval] backend   : vLLM (port {port})")
    else:
        n_workers = args.workers or N_WORKERS_OLLAMA
        classify_fn = lambda text, img: classify_meme_ollama(args.model, text, img)
        print(f"[eval] backend   : Ollama")

    if not args.force and not args.dry_run:
        evaluated = get_evaluated(RUNS_DIR_MM, args.dataset)
        if args.model in evaluated:
            print(f"[eval] {args.model} ya evaluado para {args.dataset}. Usa --force.")
            return

    test_set = load_dataset(ds_cfg)
    if args.dry_run:
        test_set = test_set[:5]

    print(f"[eval] modelo    : {args.model}")
    print(f"[eval] familia   : {model_meta.get('family')}  |  {model_meta.get('params')}  |  {model_meta.get('size_gb')} GB")
    print(f"[eval] dataset   : {args.dataset} - {ds_cfg['display']}")
    print(f"[eval] instancias: {len(test_set)}")
    print(f"[eval] workers   : {n_workers}")
    print(f"[eval] runs dir  : {RUNS_DIR_MM}")
    print(f"[eval] pred dir  : {PRED_DIR_MM}")

    results_map     = {}
    checkpoint_lock = threading.Lock()
    partial_path    = RUNS_DIR_MM / f"partial_{safe_model}_{args.dataset}.jsonl"
    completed       = 0
    t0              = time.time()

    pbar = tqdm(total=len(test_set), desc=f"{args.model} x {args.dataset}", unit="meme")

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(classify_fn, item["text"], item["image_path"]): i
            for i, item in enumerate(test_set)
        }
        for future in as_completed(futures):
            i                = futures[future]
            pred, rat, raw   = future.result()
            results_map[i]   = (pred, rat, raw)
            completed       += 1
            pbar.update(1)

            if not args.dry_run and completed % SAVE_EVERY == 0:
                with checkpoint_lock:
                    save_checkpoint(partial_path, results_map, test_set, args.model, args.dataset)
                tqdm.write(f"  [checkpoint] {completed}/{len(test_set)}")

    pbar.close()
    elapsed = time.time() - t0

    trues, preds, predictions_rows, n_unclear = [], [], [], 0
    for i, item in enumerate(test_set):
        pred, rat, raw = results_map.get(i, ("unclear", "", ""))
        if pred == "unclear":
            n_unclear += 1
        trues.append(item["label"])
        preds.append(pred)
        predictions_rows.append({
            "idx": i, "text": item["text"], "image_path": item["image_path"],
            "gold_label": item["label"], "pred_label": pred, "rationale": rat,
            "raw_response": raw, "model": args.model, "dataset": args.dataset,
        })

    if args.dry_run:
        for row in predictions_rows:
            print(f"\n[{row['gold_label']} → {row['pred_label']}] {row['text'][:60]}")
            print(f"  rationale: {row['rationale']}")
            print(f"  raw: {row['raw_response'][:120]}")
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
    print(f"[eval] tiempo              : {elapsed:.0f}s ({elapsed/len(test_set):.1f}s/meme)")

    pred_fname = f"{ts.strftime('%Y-%m-%d')}_{safe_model}_{args.dataset}_predictions.jsonl"
    fname      = f"{ts.strftime('%Y-%m-%d')}_{safe_model}_{args.dataset}.json"

    result = {
        "run_id": fname, "timestamp": ts.isoformat(),
        "model": args.model, "family": model_meta.get("family", ""),
        "params": model_meta.get("params", ""), "size_gb": model_meta.get("size_gb"),
        "backend": backend,
        "dataset": args.dataset, "dataset_display": ds_cfg["display"],
        "modality": "image+text",
        "n_instances": len(test_set), "n_unclear": n_unclear, "n_workers": n_workers,
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
        "prompt_hash": "v2_multimodal_rationale",
        "predictions_file": f"results/predictions_multimodal/{pred_fname}",
    }

    (RUNS_DIR_MM / fname).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[eval] saved -> results/runs_multimodal/{fname}")

    with (PRED_DIR_MM / pred_fname).open("w", encoding="utf-8") as pf:
        for row in predictions_rows:
            pf.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[eval] saved -> results/predictions_multimodal/{pred_fname}")

    if partial_path.exists():
        partial_path.unlink()


if __name__ == "__main__":
    main()
