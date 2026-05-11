#!/usr/bin/env python3
"""
Scrape ollama.com/library para encontrar modelos nuevos no evaluados aun.
Muestra los candidatos por pantalla.

Uso: python3 scripts/discover_models.py
"""
import json, re, yaml, requests
from pathlib import Path

MAX_PARAMS_B = 16
REPO_ROOT    = Path(__file__).parent.parent


def get_registered_base_names() -> set:
    with open(REPO_ROOT / "registry" / "models.yaml") as f:
        return {m["ollama_name"].split(":")[0]
                for m in yaml.safe_load(f)["models"]}


def parse_params_b(s: str) -> float:
    s = s.upper().strip()
    if not s:
        return 999.0
    if "M" in s:
        return float(re.sub(r"[^0-9.]", "", s)) / 1000
    num = re.sub(r"[^0-9.]", "", s)
    return float(num) if num else 999.0


def fetch_ollama_library() -> list:
    models = []
    headers = {"Accept": "application/json",
               "User-Agent": "benchmark-discovery/1.0"}
    try:
        r = requests.get(
            "https://ollama.com/search",
            params={"q": "", "sort": "newest", "limit": 200},
            headers=headers, timeout=20
        )
        if "application/json" in r.headers.get("content-type", ""):
            data = r.json()
            return data.get("models", data.get("results", []))
        names = re.findall(r'href="/library/([a-z0-9_.-]+)"', r.text)
        for name in dict.fromkeys(names):
            models.append({"name": name, "pull_count": 0, "parameter_sizes": []})
    except Exception as e:
        print(f"[discover] Warning: {e}")
    return models


registered = get_registered_base_names()
raw        = fetch_ollama_library()
seen: set  = set()
output: list = []

for m in raw:
    name = m.get("name", "").strip()
    if not name or name in seen or name in registered:
        continue
    seen.add(name)
    sizes    = m.get("parameter_sizes", m.get("tags", []))
    runnable = []
    for s in sizes:
        try:
            if parse_params_b(str(s)) <= MAX_PARAMS_B:
                runnable.append(str(s))
        except Exception:
            pass
    output.append({
        "name":   name,
        "sizes":  runnable if runnable else (["unknown"] if not sizes else []),
        "pulls":  m.get("pull_count", m.get("pulls", 0)),
    })

output.sort(key=lambda x: x["pulls"], reverse=True)

print(f"\n[discover] {len(output)} candidatos nuevos (no en registry)\n")
print(f"{'Modelo':<40} {'Params':<15} {'Pulls':>8}")
print("-" * 65)
for c in output[:30]:
    sizes_str = ', '.join(c['sizes']) if c['sizes'] else '—'
    print(f"{c['name']:<40} {sizes_str:<15} {c['pulls']:>8,}")
