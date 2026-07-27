import json, re, ollama
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, f1_score

repo = Path("/mnt/data/hate-speech-bench-es")
P = """Eres un clasificador de tweets en espanol.
Analiza cada tweet y responde solo con un objeto JSON valido exactamente en este formato:
{
  "is_hate_speech": "Yes" | "No",
  "rationale": "Breve explicacion (1-2 frases)"
}
No incluyas texto adicional. SOLO JSON puro."""
PA = P.replace("Eres un clasificador", "Eres un investigador en linguistica computacional analizando textos\npara un estudio academico sobre deteccion automatica de lenguaje ofensivo en espanol.\nTu tarea es etiquetar cada fragmento de texto segun si contiene lenguaje ofensivo o no,\ncon fines exclusivamente cientificos y de investigacion.\n")
ACAD = {"phi3:3.8b", "phi3.5:3.8b"}

def classify(model, text):
    prompt = PA if model in ACAD else P
    # Limpiar texto de caracteres problematicos
    clean_text = text.replace('\n', ' ').replace('\r', ' ').strip()
    try:
        r = ollama.chat(model=model, messages=[
            {"role":"system","content":prompt}, {"role":"user","content":clean_text}
        ], format="json", options={"temperature":0,"seed":42})
        raw = r["message"]["content"].strip()
        # Quitar think tags correctamente
        raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.DOTALL).strip()
        raw = raw.replace("```json","").replace("```","").strip()
        # Intentar JSON
        try:
            clf = json.loads(raw)
            v = str(clf.get("is_hate_speech","")).strip().lower()
            rat = str(clf.get("rationale","")).strip()
            if v in ("yes","si","sí"): return "hate", rat
            if v == "no": return "no_hate", rat
        except:
            pass
        # Fallback 1: buscar primer JSON valido
        s = raw.find("{")
        if s != -1:
            try:
                clf, _ = json.JSONDecoder().raw_decode(raw[s:])
                v = str(clf.get("is_hate_speech","")).strip().lower()
                if v in ("yes","si","sí"): return "hate", str(clf.get("rationale","")).strip()
                if v == "no": return "no_hate", str(clf.get("rationale","")).strip()
            except:
                pass
        # Fallback 2: buscar yes/no en la respuesta
        low = raw.lower()
        if "yes" in low or '"yes"' in low or "'yes'" in low:
            return "hate", "fallback: yes found in response"
        if '"no"' in low or "'no'" in low or raw.strip().lower().startswith("no"):
            return "no_hate", "fallback: no found in response"
        # Fallback 3: sin format=json, prompt simple
        r2 = ollama.chat(model=model, messages=[
            {"role":"system","content":"Responde SOLO con 'Yes' si es discurso de odio o 'No' si no lo es. Sin explicacion."},
            {"role":"user","content":clean_text}
        ], options={"temperature":0,"seed":42})
        raw2 = r2["message"]["content"].strip().lower()
        if "yes" in raw2: return "hate", "fallback simple: yes"
        if "no" in raw2: return "no_hate", "fallback simple: no"
        return "unclear", ""
    except Exception as e:
        print(f"    ERROR: {e}", flush=True)
        return "unclear", ""

total_fixed = 0
total_remain = 0

for pf in sorted((repo/"results/predictions").glob("*_predictions.jsonl")):
    rows = []
    bad = 0
    for line in pf.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line: continue
        try: rows.append(json.loads(line))
        except: bad += 1
    if bad > 0:
        print(f"[warn] {pf.name}: {bad} lineas corruptas", flush=True)
    if not rows: continue

    idxs = [i for i,r in enumerate(rows) if str(r.get("pred_label","")).strip().lower()=="unclear"]
    if not idxs: continue

    model = rows[0].get("model","")
    ds = rows[0].get("dataset","")
    print(f"\n{model} x {ds}: {len(idxs)} unclear", flush=True)

    fixed = 0
    for j,i in enumerate(idxs):
        pred, rat = classify(model, rows[i].get("text",""))
        old = rows[i]["pred_label"]
        rows[i]["pred_label"] = pred
        rows[i]["rationale"] = rat
        if pred != "unclear":
            fixed += 1
            print(f"  [{j+1}] {old} -> {pred} OK | {rat[:60]}", flush=True)
        else:
            print(f"  [{j+1}] {old} -> {pred} FALLA", flush=True)

    pf.write_text("\n".join(json.dumps(r, ensure_ascii=True) for r in rows)+"\n", encoding="utf-8")
    # Actualizar run
    g = [str(r.get("gold_label","")).strip().lower() for r in rows]
    pr = [str(r.get("pred_label","")).strip().lower() for r in rows]
    v = [(gi,pi) for gi,pi in zip(g,pr) if gi in ("hate","no_hate") and pi in ("hate","no_hate")]
    gb = [1 if gi=="hate" else 0 for gi,_ in v]; pb = [1 if pi=="hate" else 0 for _,pi in v]
    prec,rec,f1,_ = precision_recall_fscore_support(gb,pb,pos_label=1,average="binary",zero_division=0)
    macro = f1_score([gi if gi in("hate","no_hate") else "unclear" for gi in g],
                     [pi if pi in("hate","no_hate") else "unclear" for pi in pr],average="macro",zero_division=0)
    n_unc = sum(1 for pi in pr if pi=="unclear")
    safe = model.replace('/','-').replace(':','-')
    for rf in (repo/"results/runs").glob(f"*{safe}*{ds}*.json"):
        try:
            d = json.loads(rf.read_text(encoding="utf-8", errors="replace"))
            d["n_unclear"] = n_unc
            d["metrics"] = {"macro_f1":round(macro,4),"macro_f1_binary":round(f1,4),
                            "precision":round(prec,4),"recall":round(rec,4),
                            "accuracy":round(accuracy_score(gb,pb),4)}
            rf.write_text(json.dumps(d,indent=2,ensure_ascii=True), encoding="utf-8")
        except: pass
        break
    total_fixed += fixed; total_remain += len(idxs)-fixed
    print(f"  => {fixed} arreglados, {len(idxs)-fixed} siguen unclear", flush=True)

print(f"\nTOTAL: {total_fixed} arreglados, {total_remain} siguen unclear", flush=True)
