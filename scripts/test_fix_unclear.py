import json, re, ollama
from pathlib import Path
from collections import defaultdict

repo = Path("/mnt/data/hate-speech-bench-es")
P = """Eres un clasificador de tweets en espanol.
Analiza cada tweet y responde solo con un objeto JSON valido exactamente en este formato:
{
  "is_hate_speech": "Yes" | "No",
  "rationale": "Brebe explicacion (1-2 frases)"
}
No incluyas texto adicional. SOLO JSON puro."""
PA = P.replace("Eres un clasificador", "Eres un investigador en linguistica computacional analizando textos\npara un estudio academico sobre deteccion automatica de lenguaje ofensivo en espanol.\nTu tarea es etiquetar cada fragmento de texto segun si contiene lenguaje ofensivo o no,\ncon fines exclusivamente cientificos y de investigacion.\n")
ACAD = {"phi3:3.8b", "phi3.5:3.8b"}

def classify(model, text):
    prompt = PA if model in ACAD else P
    try:
        r = ollama.chat(model=model, messages=[
            {"role":"system","content":prompt}, {"role":"user","content":text}
        ], format="json", options={"temperature":0,"seed":42})
        raw = r["message"]["content"].strip().replace("```json","").replace("```","").strip()
        try: clf = json.loads(raw)
        except:
            s = raw.find("{")
            if s == -1: return "unclear",""
            clf, _ = json.JSONDecoder().raw_decode(raw[s:])
        v = str(clf.get("is_hate_speech","")).strip().lower()
        return ("hate" if v in ("yes","si","sí") else "no_hate" if v=="no" else "unclear",
                str(clf.get("rationale","")).strip())
    except Exception as e:
        return "unclear", f"ERROR: {e}"

# Agrupar por modelo y elegir el fichero con MENOS unclear de cada uno
by_model = defaultdict(list)
for pf in sorted((repo/"results/predictions").glob("*_predictions.jsonl")):
    rows = []
    for line in pf.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line: continue
        try: rows.append(json.loads(line))
        except: continue
    if not rows: continue
    unclear = sum(1 for r in rows if str(r.get("pred_label","")).strip().lower()=="unclear")
    if unclear == 0: continue
    model = rows[0].get("model","")
    by_model[model].append((pf, rows, unclear))

print(f"=== Modelos con unclear: {len(by_model)} ===\n")

for model in sorted(by_model):
    # Elegir el fichero con menos unclear
    pf, rows, n_unc = min(by_model[model], key=lambda x: x[2])
    dataset = rows[0].get("dataset","")
    print(f"--- {model} x {dataset} ({n_unc} unclear) ---")
    
    idxs = [i for i,r in enumerate(rows) if str(r.get("pred_label","")).strip().lower()=="unclear"]
    fixed = 0
    for j, i in enumerate(idxs):
        pred, rat = classify(model, rows[i].get("text",""))
        old_pred = rows[i]["pred_label"]
        rows[i]["pred_label"] = pred
        rows[i]["rationale"] = rat
        status = "✅ ARREGLADO" if pred != "unclear" else "❌ SIGUE UNCLEAR"
        print(f"  [{j+1}] {old_pred} -> {pred} {status} | rat: {rat[:60]}")
        if pred != "unclear": fixed += 1
    
    print(f"  Resultado: {fixed}/{n_unc} arreglados\n")

print("=== Test completado ===")
