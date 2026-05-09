#!/usr/bin/env python3
"""
Extrae los 5 datasets del CSV unificado es_hf_102024.csv
y los guarda como ficheros CSV individuales en data/.

Uso:
    python3 scripts/extract_datasets.py
"""
import pandas as pd
from pathlib import Path

SRC  = Path(__file__).parent.parent / "data" / "es_hf_102024.csv"
OUT  = Path(__file__).parent.parent / "data"

DATASETS = ["hateval", "haternet", "chileno", "hascosva", "misocorpus"]

df = pd.read_csv(SRC)
print(f"CSV total: {len(df)} filas\n")

for ds in DATASETS:
    subset = df[df["dataset"] == ds][["text", "labels", "source", "tweet_id"]].copy()
    subset["labels"] = subset["labels"].astype(int)

    out_path = OUT / f"{ds}_test.csv"
    subset.to_csv(out_path, index=False)

    n_hate    = (subset["labels"] == 1).sum()
    n_no_hate = (subset["labels"] == 0).sum()
    print(f"{ds:<12} → {len(subset):>5} filas  |  hate: {n_hate}  no_hate: {n_no_hate}  → {out_path.name}")

print("\nListo. Ficheros guardados en data/")
