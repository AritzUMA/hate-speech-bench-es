#!/usr/bin/env python3
"""
Lee results/runs/*.json y genera docs/index.html.
Uso: python3 scripts/generate_dashboard.py
"""
import json, yaml
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

REPO_ROOT = Path(__file__).parent.parent
RUNS_DIR  = REPO_ROOT / "results" / "runs"
DOCS_DIR  = REPO_ROOT / "docs"


def load_runs():
    runs = []
    for f in sorted(RUNS_DIR.glob("*.json")):
        if f.name.startswith("partial_"):
            continue
        try:
            runs.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception as e:
            print(f"  [warn] skip {f.name}: {e}")
    return runs


def load_registry():
    with open(REPO_ROOT / "registry" / "models.yaml") as f:
        models = {m["ollama_name"]: m for m in yaml.safe_load(f)["models"]}
    with open(REPO_ROOT / "registry" / "datasets.yaml") as f:
        datasets = {d["name"]: d for d in yaml.safe_load(f)["datasets"]}
    return models, datasets


def build_index(runs, models_meta, datasets_meta):
    best = {}
    for r in sorted(runs, key=lambda x: x["timestamp"]):
        key = (r["model"], r["dataset"])
        best[key] = r

    results = defaultdict(dict)
    history = []

    for (model, dataset), r in best.items():
        m = r["metrics"]
        # Override family/params from registry (run JSON may have stale values)
        meta = models_meta.get(model, {})
        results[model][dataset] = {
            "macro_f1":        m.get("macro_f1", 0),
            "macro_f1_binary": m.get("macro_f1_binary", m.get("macro_f1", 0)),
            "precision":       m.get("precision", 0),
            "recall":          m.get("recall", 0),
            "accuracy":        m.get("accuracy", 0),
            "hate_f1":         r["per_class"].get("hate", {}).get("f1-score", 0),
            "n_instances":     r.get("n_instances", 0),
            "n_unclear":       r.get("n_unclear", r.get("n_sin_pred", 0)),
            "coverage":        round((r.get("n_instances",0) - r.get("n_unclear", r.get("n_sin_pred",0))) / r.get("n_instances",1) * 100, 1),
            "elapsed":         r.get("elapsed_seconds"),
            "timestamp":       r["timestamp"],
        }
        history.append({
            "ts":      r["timestamp"][:10],
            "model":   model,
            "dataset": dataset,
            "f1":      m.get("macro_f1", 0),
            "hate_f1": r["per_class"].get("hate", {}).get("f1-score", 0),
            "elapsed": r.get("elapsed_seconds"),
        })

    history.sort(key=lambda x: x["ts"], reverse=True)
    active_models   = sorted(results.keys())
    active_datasets = sorted({d for ds in results.values() for d in ds.keys()})
    n_datasets      = len(active_datasets)

    METRICS = ["macro_f1", "macro_f1_binary", "hate_f1", "precision", "recall", "accuracy"]
    overall = {}
    for model, ds_scores in results.items():
        # Ensure family comes from registry
        vals = {m: [] for m in METRICS}
        for ds_name, scores in ds_scores.items():
            for m in METRICS:
                v = scores.get(m)
                if v is not None:
                    vals[m].append(v)
        overall[model] = {m: round(sum(v)/len(v), 4) if v else None for m, v in vals.items()}
        overall[model]["n_datasets"]       = len(ds_scores)
        overall[model]["n_datasets_total"] = n_datasets
        n_u = sum(ds_scores[d].get("n_unclear", 0) for d in ds_scores)
        n_i = sum(ds_scores[d].get("n_instances", 0) for d in ds_scores)
        overall[model]["coverage"] = round((n_i - n_u) / n_i * 100, 1) if n_i else 100

    best_f1 = max((v["macro_f1"] or 0 for v in overall.values()), default=0)

    return {
        "generated":       datetime.now(timezone.utc).isoformat(),
        "n_runs":          len(best),
        "n_models":        len(active_models),
        "n_datasets":      n_datasets,
        "best_overall_f1": round(best_f1, 3),
        "models":          active_models,
        "datasets":        active_datasets,
        "results":         dict(results),
        "overall":         overall,
        "history":         history[:100],
        "models_meta":     {k: {
            "display_name": v.get("display_name", k),
            "family":       v.get("family", ""),
            "params":       v.get("params", ""),
            "params_exact": v.get("params_exact"),
            "size_gb":      v.get("size_gb"),
            "release_date": v.get("release_date", ""),
            "developer":    v.get("developer", ""),
        } for k, v in models_meta.items() if k in active_models},
        "datasets_meta":   {k: {
            "display":  v.get("display", k),
            "platform": v.get("platform", ""),
            "source":   v.get("source", ""),
            "n_total":  v.get("n_total", 0),
            "hate":     v.get("hate", 0),
            "no_hate":  v.get("no_hate", 0),
            "iaa":      v.get("iaa", ""),
            "topic":    v.get("topic", ""),
            "lang":     v.get("lang", ""),
            "paper":    v.get("paper", ""),
        } for k, v in datasets_meta.items() if k in active_datasets},
    }


def generate_html(index):
    index_json = json.dumps(index, ensure_ascii=False)
    return """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Deteccion de Discurso de Odio en Espanol - Benchmark Continuo</title>
<script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
<script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
<script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;font-size:14px;background:#f9f9f7;color:#1a1a1a;line-height:1.6}
.wrap{max-width:1200px;margin:0 auto;padding:2.5rem 1.5rem}
.gh-btn{position:fixed;top:14px;right:18px;display:flex;align-items:center;gap:7px;background:#fff;border:0.5px solid #d0d0d0;border-radius:8px;padding:6px 12px;font-size:12px;color:#333;text-decoration:none;z-index:1000;box-shadow:0 1px 4px rgba(0,0,0,.08);transition:box-shadow .15s}
.gh-btn:hover{box-shadow:0 2px 8px rgba(0,0,0,.15);color:#000}
.header{margin-bottom:3rem;padding-bottom:2rem;border-bottom:0.5px solid #e0e0e0}
.header h1{font-size:24px;font-weight:600;margin-bottom:.4rem;letter-spacing:-.3px;display:flex;align-items:center;gap:8px}
.live{display:inline-block;width:8px;height:8px;border-radius:50%;background:#2e7d32;animation:pulse 2s infinite;flex-shrink:0}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.header-meta{font-size:12px;color:#aaa;margin-bottom:1rem}
.header-desc{font-size:14px;color:#444;max-width:860px;margin-bottom:1rem;line-height:1.75}
.header-method{font-size:13px;color:#666;margin-bottom:1rem;line-height:1.8}
.section{margin-bottom:3.5rem}
.section-header{display:flex;align-items:baseline;gap:12px;margin-bottom:1rem;padding-bottom:.5rem;border-bottom:0.5px solid #eee}
.section-header h2{font-size:16px;font-weight:600}
.section-sub{font-size:12px;color:#aaa}
.chart-wrap{background:#fff;border:0.5px solid #e0e0e0;border-radius:10px;padding:1.2rem 1.4rem;margin-bottom:1.5rem}
.chart-label{font-size:12px;color:#888;margin-bottom:.6rem}
.chart-box{position:relative;height:420px}
.chart-box-scatter{position:relative;height:420px}
table{width:100%;border-collapse:collapse;font-size:13px;background:#fff;border-radius:10px;overflow:hidden;border:0.5px solid #e0e0e0}
th{padding:9px 12px;text-align:left;font-weight:500;font-size:12px;color:#666;background:#fafaf8;border-bottom:0.5px solid #e8e8e8;cursor:pointer;user-select:none;white-space:nowrap}
th:hover{background:#f0f0ee}
td{padding:8px 12px;border-bottom:0.5px solid #f0f0f0;vertical-align:middle}
tr:last-child td{border-bottom:none}
tr:hover td{background:#fafaf8}
.bar-wrap{display:flex;align-items:center;gap:8px}
.bar-bg{flex:1;height:5px;background:#eee;border-radius:3px;min-width:50px}
.bar-fill{height:5px;border-radius:3px}
.tag{font-size:10px;padding:2px 7px;border-radius:12px;background:#eef;color:#336;white-space:nowrap}
.tag-ds{font-size:10px;padding:2px 7px;border-radius:12px;background:#fef3e2;color:#b45309}
.chip{padding:4px 12px;border:0.5px solid #ddd;border-radius:20px;font-size:12px;cursor:pointer;background:#fff;color:#555;transition:all .15s}
.chip:hover{border-color:#999;color:#111}
.chip.active{background:#1a1a1a;color:#fff;border-color:#1a1a1a;font-weight:500}
.chips{display:flex;gap:6px;flex-wrap:wrap}
.controls{display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:10px;margin-bottom:12px}
.r1{color:#B8860B;font-weight:600}
.r2{color:#71717a;font-weight:500}
.r3{color:#6366f1}
.rank{color:#bbb;font-size:12px}
select{font-size:12px;padding:5px 9px;border-radius:7px;border:0.5px solid #ddd;background:#fff;cursor:pointer}
.cov-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:4px}
.info-box{background:#f0f4ff;border:0.5px solid #c7d2fe;border-radius:8px;padding:.8rem 1rem;font-size:12px;color:#4338ca;margin-bottom:1rem}
</style>
</head>
<body>
<div id="root"></div>
<script>window.__INDEX__ = """ + index_json + """;</script>
<script type="text/babel">
const { useState, useMemo, useEffect, useRef } = React;
const DATA = window.__INDEX__;

const CDN   = 'https://unpkg.com/@lobehub/icons-static-png@latest/light/';
const LOGOS = 'https://aritzuma.github.io/hate-speech-bench-es/logos/';
const FAMILY_ICON = {
  // Texto
  'LLaMA 3.1':   CDN+'meta.png',
  'LLaMA':       CDN+'meta.png',
  'LLaMA 3.2':   CDN+'meta.png',
  'Llama':       CDN+'meta.png',
  'Gemma 2':     CDN+'gemma.png',
  'Gemma 3':     CDN+'gemma.png',
  'Gemma 4':     CDN+'gemma.png',
  'Gemma':       CDN+'gemma.png',
  'Qwen 2.5':    CDN+'qwen.png',
  'Qwen 3':      CDN+'qwen.png',
  'Qwen 3.5':    CDN+'qwen.png',
  'Qwen':        CDN+'qwen.png',
  'Mistral':     CDN+'mistral.png',
  'DeepSeek R1': CDN+'deepseek.png',
  'Phi 3':       CDN+'microsoft.png',
  'Phi 3.5':     CDN+'microsoft.png',
  'Phi':         CDN+'microsoft.png',
  'Yi':          CDN+'yi.png',
  'StableLM 2':  CDN+'stability-ai.png',
  'GPT-OSS':     CDN+'openai.png',
  'Salamandra':  LOGOS+'salamandra.png',
  'OpenEuroLLM': LOGOS+'openeurollm.png',
  // Vision
  'Moondream':   LOGOS+'moondream.png',
  'LLaVA':       CDN+'meta.png',
  'InternVL':    CDN+'internlm.png',
  'MiniCPM':     CDN+'minicpm.png',
  'Granite':     CDN+'ibm.png',
};

const COLORS = {
  'LLaMA 3.1':'#3b82f6','LLaMA':'#3b82f6',
  'Gemma 2':'#10b981','Gemma 3':'#059669','Gemma 4':'#047857',
  'Qwen 2.5':'#f59e0b','Qwen 3':'#d97706','Qwen 3.5':'#b45309',
  'Mistral':'#8b5cf6','DeepSeek R1':'#ec4899',
  'Salamandra':'#f97316','GPT-OSS':'#6b7280',
  'Yi':'#6366f1','Phi 3.5':'#14b8a6','Phi 3':'#0d9488',
  'StableLM 2':'#9ca3af','OpenEuroLLM':'#84cc16',
  // Vision
  'LLaMA 3.2':'#3b82f6','Llama':'#3b82f6',
  'Phi':'#14b8a6',
  'Gemma':'#10b981',
  'Qwen':'#f59e0b',
  'LLaVA':'#f97316','InternVL':'#6366f1',
  'MiniCPM':'#06b6d4','Granite':'#334155',
  'Salamandra':'#f97316','Moondream':'#8b5cf6',
};
const MONTHS = ['Ene','Feb','Mar','Abr','May','Jun','Jul','Ago','Sep','Oct','Nov','Dic'];
const METRICS = [
  {key:'macro_f1',        label:'Macro-F1'},
  {key:'macro_f1_binary', label:'Macro-F1 (binario)'},
  {key:'hate_f1',         label:'Hate F1'},
  {key:'precision',       label:'Precision'},
  {key:'recall',          label:'Recall'},
  {key:'accuracy',        label:'Accuracy'},
];
const col = f => COLORS[f] || '#6b7280';
const fmt = v => v != null ? v.toFixed(4) : '-';

const loadedImgs = {};
Object.entries(FAMILY_ICON).forEach(([fam, url]) => {
  if (loadedImgs[fam]) return;
  const img = new Image();
  img.src = url;
  loadedImgs[fam] = img;
});

const logoPlugin = {
  id: 'logoPlugin',
  afterDraw(chart) {
    const ctx = chart.ctx;
    const meta = chart.getDatasetMeta(0);
    const fams = chart.data.datasets[0]._families || [];
    const size = 18;
    meta.data.forEach((bar, i) => {
      const fam = fams[i];
      const img = loadedImgs[fam];
      if (!img || !img.complete || !img.naturalWidth) return;
      ctx.drawImage(img, bar.x - size/2, bar.y - size - 5, size, size);
    });
  }
};
Chart.register(logoPlugin);

const GH_ICON = (
  React.createElement('svg', {width:15, height:15, viewBox:'0 0 16 16', fill:'currentColor'},
    React.createElement('path', {d:'M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z'})
  )
);

function SortTh({k, label, sortKey, sortDir, onSort}) {
  return (
    <th onClick={() => onSort(k)}>
      {label}
      <span style={{color:'#ccc', marginLeft:3}}>
        {sortKey===k ? (sortDir==='desc' ? 'v' : '^') : '-'}
      </span>
    </th>
  );
}

function Leaderboard({ds, setDs}) {
  const [sortKey, setSortKey] = useState('macro_f1');
  const [sortDir, setSortDir] = useState('desc');

  function onSort(k) {
    if (k === sortKey) setSortDir(d => d==='desc' ? 'asc' : 'desc');
    else { setSortKey(k); setSortDir('desc'); }
  }

  const isOv = ds === '__overall__';

  const rows = useMemo(() => {
    const r = DATA.models
      .filter(m => isOv ? DATA.overall[m] : DATA.results[m] && DATA.results[m][ds])
      .map(m => ({
        model:        m,
        display_name: DATA.models_meta[m] ? DATA.models_meta[m].display_name : m,
        family:       DATA.models_meta[m] ? DATA.models_meta[m].family : '',
        params:       DATA.models_meta[m] ? DATA.models_meta[m].params : '',
        developer:    DATA.models_meta[m] ? DATA.models_meta[m].developer : '',
        release_date: DATA.models_meta[m] ? DATA.models_meta[m].release_date : '',
        ...(isOv ? DATA.overall[m] : DATA.results[m][ds]),
      }));
    r.sort((a, b) => sortDir==='desc'
      ? (b[sortKey] != null ? b[sortKey] : -1) - (a[sortKey] != null ? a[sortKey] : -1)
      : (a[sortKey] != null ? a[sortKey] : -1) - (b[sortKey] != null ? b[sortKey] : -1));
    return r;
  }, [ds, sortKey, sortDir]);

  const maxVal = Math.max(...rows.map(r => r[sortKey] || 0));

  return (
    <div className="section">
      <div className="section-header">
        <h2>Leaderboard</h2>
        <span className="section-sub">{DATA.n_models} modelos - {DATA.n_datasets} datasets</span>
      </div>
      <div className="controls">
        <div className="chips">
          <button className={'chip' + (isOv ? ' active' : '')} onClick={() => setDs('__overall__')}>
            Overall
          </button>
          {DATA.datasets.map(d => (
            <button key={d} className={'chip' + (d===ds ? ' active' : '')} onClick={() => setDs(d)}>
              {DATA.datasets_meta[d] ? DATA.datasets_meta[d].display : d}
            </button>
          ))}
        </div>
        <select value={sortKey} onChange={e => { setSortKey(e.target.value); setSortDir('desc'); }}>
          {METRICS.map(m => <option key={m.key} value={m.key}>{m.label}</option>)}
        </select>
      </div>
      {isOv && (
        <div className="info-box">
          Media de cada metrica sobre todos los datasets evaluados por modelo.
          Macro-F1 binario excluye predicciones unclear.
          DS indica cuantos datasets tiene evaluados cada modelo.
        </div>
      )}
      <table>
        <thead>
          <tr>
            <th style={{width:28}}>#</th>
            <th>Modelo</th>
            <th>Familia</th>
            <th>Params</th>
            {isOv && <th>Developer</th>}
            {isOv && <th>Release</th>}
            <SortTh k="macro_f1"        label="Macro-F1"   sortKey={sortKey} sortDir={sortDir} onSort={onSort}/>
            <SortTh k="macro_f1_binary" label="F1 binario" sortKey={sortKey} sortDir={sortDir} onSort={onSort}/>
            <SortTh k="hate_f1"         label="Hate F1"    sortKey={sortKey} sortDir={sortDir} onSort={onSort}/>
            <SortTh k="accuracy"        label="Accuracy"   sortKey={sortKey} sortDir={sortDir} onSort={onSort}/>
            <SortTh k="precision"       label="Precision"  sortKey={sortKey} sortDir={sortDir} onSort={onSort}/>
            <SortTh k="recall"          label="Recall"     sortKey={sortKey} sortDir={sortDir} onSort={onSort}/>
            <th>{isOv ? 'DS' : 'n'}</th>
            <th>Cobertura</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const pct      = maxVal > 0 ? (r[sortKey] || 0) / maxVal * 100 : 0;
            const rk       = i===0 ? 'r1' : i===1 ? 'r2' : i===2 ? 'r3' : 'rank';
            const cov      = r.coverage;
            const covColor = cov == null ? '#aaa' : cov >= 95 ? '#10b981' : cov >= 80 ? '#f59e0b' : '#ef4444';
            const iconUrl  = FAMILY_ICON[r.family];
            return (
              <tr key={r.model}>
                <td className={rk}>{i+1}</td>
                <td>
                  {iconUrl
                    ? <img src={iconUrl} width={16} height={16}
                        style={{marginRight:6, verticalAlign:'middle', borderRadius:2}}
                        onError={e => { e.target.style.display='none'; }}/>
                    : null}
                  <b>{r.display_name || r.model}</b>
                </td>
                <td style={{color: col(r.family), fontWeight:500}}>{r.family}</td>
                <td><span className="tag">{r.params}</span></td>
                {isOv && <td style={{color:'#888', fontSize:12}}>{r.developer}</td>}
                {isOv && <td style={{color:'#888', fontSize:12}}>{r.release_date}</td>}
                <td>
                  <div className="bar-wrap">
                    <div className="bar-bg">
                      <div className="bar-fill" style={{width: pct+'%', background: col(r.family)}}/>
                    </div>
                    <span style={{minWidth:44, textAlign:'right', fontVariantNumeric:'tabular-nums'}}>
                      {fmt(r.macro_f1)}
                    </span>
                  </div>
                </td>
                <td>{fmt(r.macro_f1_binary)}</td>
                <td>{fmt(r.hate_f1)}</td>
                <td>{fmt(r.accuracy)}</td>
                <td>{fmt(r.precision)}</td>
                <td>{fmt(r.recall)}</td>
                <td style={{color:'#aaa'}}>
                  {isOv
                    ? <span className="tag-ds">{r.n_datasets}/{r.n_datasets_total}</span>
                    : (r.n_instances ? r.n_instances.toLocaleString() : '-')}
                </td>
                <td>
                  <span className="cov-dot" style={{background: covColor}}/>
                  {cov != null ? cov + '%' : '-'}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function HateF1Bar({ds}) {
  const ref = useRef(null);

  const chartData = useMemo(() => {
    const isOv  = ds === '__overall__';
    const getF1 = m => isOv ? (DATA.overall[m]?.hate_f1||0) : (DATA.results[m]?.[ds]?.hate_f1||0);
    const models = DATA.models.filter(m => isOv ? DATA.overall[m] : DATA.results[m]?.[ds]);
    const sorted = [...models].sort((a, b) => getF1(b) - getF1(a));
    return {
      labels:   sorted.map(m => DATA.models_meta[m] ? DATA.models_meta[m].display_name : m),
      values:   sorted.map(m => getF1(m)),
      colors:   sorted.map(m => col(DATA.models_meta[m] ? DATA.models_meta[m].family : '')),
      families: sorted.map(m => DATA.models_meta[m] ? DATA.models_meta[m].family : ''),
    };
  }, [ds]);

  useEffect(() => {
    if (!ref.current) return;
    const ex = Chart.getChart(ref.current);
    if (ex) ex.destroy();
    const dataset = {
      data:            chartData.values,
      backgroundColor: chartData.colors,
      borderRadius:    4,
      borderSkipped:   false,
      _families:       chartData.families,
    };
    new Chart(ref.current, {
      type: 'bar',
      data: { labels: chartData.labels, datasets: [dataset] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: { top: 28 } },
        plugins: {
          legend: {display: false},
          tooltip: {callbacks: {label: ctx => ' Hate F1: ' + ctx.raw.toFixed(4)}}
        },
        scales: {
          x: {
            grid: {display: false},
            ticks: {font:{size:10}, color:'#444', maxRotation:35, minRotation:25}
          },
          y: {
            min: 0, max: 1,
            grid: {color:'#f0f0f0'},
            ticks: {font:{size:11}, color:'#888'}
          }
        }
      }
    });
  }, [ds]);

  return <div className="chart-box"><canvas ref={ref}/></div>;
}

// Plugin tipo ggrepel: evita solapamiento entre etiquetas y puntos
const leaderLabelPlugin = {
  id: 'leaderLabel',

  afterDatasetsDraw(chart) {
    const ctx  = chart.ctx;
    const area = chart.chartArea;

    const FONT      = 10;
    const PAD_X     = 5;
    const PAD_Y     = 3;
    const LABEL_GAP = 5;
    const POINT_GAP = 8;

    const clamp = (v, min, max) => Math.max(min, Math.min(max, v));

    ctx.save();
    ctx.font         = `${FONT}px system-ui,sans-serif`;
    ctx.textBaseline = 'middle';

    const nodes = [];
    const offsets = [
      [46, -34], [-46, -34], [46, 34], [-46, 34],
      [70, -10], [-70, -10], [70, 10], [-70, 10],
      [0, -58],  [0, 58],
      [92, -42], [-92, -42], [92, 42], [-92, 42],
    ];

    let idx = 0;

    chart.data.datasets.forEach((ds, di) => {
      const meta = chart.getDatasetMeta(di);
      if (meta.hidden) return;
      ds.data.forEach((pt, pi) => {
        const el = meta.data[pi];
        if (!el || !pt.label) return;
        const label = String(pt.label);
        const w = ctx.measureText(label).width + PAD_X * 2;
        const h = FONT + PAD_Y * 2;
        const [ox, oy] = offsets[idx % offsets.length];
        const x0 = el.x + ox;
        const y0 = el.y + oy;
        nodes.push({
          id: idx, px: el.x, py: el.y,
          tx: x0, ty: y0, x: x0, y: y0,
          w, h, label,
          color: ds.borderColor || '#999',
          pointR: ds.pointRadius || 7,
        });
        idx += 1;
      });
    });

    if (!nodes.length) { ctx.restore(); return; }

    function clampNode(n) {
      n.x = clamp(n.x, area.left + n.w/2 + 2, area.right  - n.w/2 - 2);
      n.y = clamp(n.y, area.top  + n.h/2 + 2, area.bottom - n.h/2 - 2);
    }

    function pushAwayFromPoint(n) {
      const halfW = n.w/2 + n.pointR + POINT_GAP;
      const halfH = n.h/2 + n.pointR + POINT_GAP;
      const dx = n.x - n.px;
      const dy = n.y - n.py;
      const overlapX = halfW - Math.abs(dx);
      const overlapY = halfH - Math.abs(dy);
      if (overlapX > 0 && overlapY > 0) {
        if (overlapX < overlapY) n.x += (dx >= 0 ? 1 : -1) * (overlapX + 1);
        else                      n.y += (dy >= 0 ? 1 : -1) * (overlapY + 1);
      }
    }

    for (let iter = 0; iter < 350; iter++) {
      nodes.forEach(n => {
        n.x += (n.tx - n.x) * 0.025;
        n.y += (n.ty - n.y) * 0.025;
        pushAwayFromPoint(n);
      });
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i+1; j < nodes.length; j++) {
          const a = nodes[i], b = nodes[j];
          const dx = a.x - b.x || 0.001;
          const dy = a.y - b.y || 0.001;
          const overlapX = (a.w+b.w)/2 + LABEL_GAP - Math.abs(dx);
          const overlapY = (a.h+b.h)/2 + LABEL_GAP - Math.abs(dy);
          if (overlapX > 0 && overlapY > 0) {
            if (overlapX < overlapY) {
              const sx = dx >= 0 ? 1 : -1;
              a.x += sx*overlapX/2; b.x -= sx*overlapX/2;
            } else {
              const sy = dy >= 0 ? 1 : -1;
              a.y += sy*overlapY/2; b.y -= sy*overlapY/2;
            }
          }
        }
      }
      nodes.forEach(n => { pushAwayFromPoint(n); clampNode(n); });
    }

    for (let iter = 0; iter < 80; iter++) {
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i+1; j < nodes.length; j++) {
          const a = nodes[i], b = nodes[j];
          const dx = a.x - b.x || 0.001;
          const dy = a.y - b.y || 0.001;
          const overlapX = (a.w+b.w)/2 + LABEL_GAP - Math.abs(dx);
          const overlapY = (a.h+b.h)/2 + LABEL_GAP - Math.abs(dy);
          if (overlapX > 0 && overlapY > 0) {
            if (overlapX < overlapY) {
              const sx = dx >= 0 ? 1 : -1;
              a.x += sx*overlapX/2; b.x -= sx*overlapX/2;
            } else {
              const sy = dy >= 0 ? 1 : -1;
              a.y += sy*overlapY/2; b.y -= sy*overlapY/2;
            }
          }
        }
      }
      nodes.forEach(n => { pushAwayFromPoint(n); clampNode(n); });
    }

    nodes.forEach(n => {
      ctx.beginPath();
      ctx.moveTo(n.px, n.py);
      ctx.lineTo(n.x, n.y);
      ctx.strokeStyle = 'rgba(0,0,0,0.18)';
      ctx.lineWidth   = 0.8;
      ctx.setLineDash([2, 3]);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = 'rgba(255,255,255,0.96)';
      ctx.fillRect(n.x - n.w/2, n.y - n.h/2, n.w, n.h);

      ctx.strokeStyle = 'rgba(0,0,0,0.08)';
      ctx.strokeRect(n.x - n.w/2, n.y - n.h/2, n.w, n.h);

      ctx.fillStyle = '#333';
      ctx.textAlign = 'center';
      ctx.fillText(n.label, n.x, n.y);
    });

    ctx.restore();
  }
};

function ReleaseDateScatter({ds}) {
  const ref = useRef(null);

  const datasets = useMemo(() => {
    const isOv  = ds === '__overall__';
    const models = DATA.models.filter(m => isOv ? DATA.overall[m] : DATA.results[m]?.[ds]);
    const byFamily = {};
    models.forEach(m => {
      const meta   = DATA.models_meta[m] || {};
      const f      = meta.family || 'Otro';
      if (!byFamily[f]) byFamily[f] = [];
      const parts  = (meta.release_date || '').split('-').map(Number);
      const y = parts[0], mo = parts[1] || 1, day = parts[2] || 1;
      if (!y) return;
      const x      = y + (mo-1)/12 + (day-1)/365;
      byFamily[f].push({ x, y: isOv?(DATA.overall[m]?.hate_f1||0):(DATA.results[m]?.[ds]?.hate_f1||0), label: (DATA.models_meta[m] ? DATA.models_meta[m].display_name : m), mo, yr: y, day });
    });
    return Object.entries(byFamily).map(([f, pts]) => ({
      label:            f,
      data:             pts,
      backgroundColor:  col(f) + 'cc',
      borderColor:      col(f),
      pointRadius:      7,
      pointHoverRadius: 9,
    }));
  }, []);

  useEffect(() => {
    if (!ref.current) return;
    const ex = Chart.getChart(ref.current);
    if (ex) ex.destroy();
    new Chart(ref.current, {
      type: 'scatter',
      data: {datasets},
      plugins: [leaderLabelPlugin],
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {display: false},
          tooltip: {callbacks: {label: ctx => {
            const mo = MONTHS[(ctx.raw.mo||1)-1];
            const day = ctx.raw.day || 1;
            return ctx.raw.label + '  (' + day + ' ' + mo + ' ' + ctx.raw.yr + ')  ' + ctx.raw.y.toFixed(4);
          }}}
        },
        layout: { padding: { top: 35, right: 45, bottom: 35, left: 45 } },
        scales: {
          x: {
            grid: {color:'#f0f0f0'},
            ticks: {
              font:{size:11}, color:'#888', maxTicksLimit:8,
              callback: function(val) {
                const y = Math.floor(val);
                const mo = Math.round((val - y) * 12);
                return MONTHS[mo >= 12 ? 11 : mo] + ' ' + y;
              }
            }
          },
          y: {
            min: 0, max: 1,
            grid: {color:'#f0f0f0'},
            ticks: {font:{size:11}, color:'#888'},
            title: {display:true, text:'Hate F1', font:{size:11}, color:'#888'}
          }
        }
      }
    });
  }, []);

  return <div className="chart-box-scatter"><canvas ref={ref}/></div>;
}

function ParamsScatter({ds}) {
  const ref = useRef(null);

  const datasets = useMemo(() => {
    const isOv  = ds === '__overall__';
    const models = DATA.models.filter(m => isOv ? DATA.overall[m] : DATA.results[m]?.[ds]);
    const byFamily = {};
    models.forEach(m => {
      const meta    = DATA.models_meta[m] || {};
      const f       = meta.family || 'Otro';
      if (!byFamily[f]) byFamily[f] = [];
      const p = meta.params_exact || parseFloat((meta.params || '0').replace('B',''));
      if (!p) return;
      const hate_f1 = isOv
        ? (DATA.overall[m]?.hate_f1 || 0)
        : (DATA.results[m]?.[ds]?.hate_f1 || 0);
      byFamily[f].push({ x: p, y: hate_f1, label: (DATA.models_meta[m] ? DATA.models_meta[m].display_name : m) });
    });
    return Object.entries(byFamily).map(([f, pts]) => ({
      label:            f,
      data:             pts,
      backgroundColor:  col(f) + 'cc',
      borderColor:      col(f),
      pointRadius:      7,
      pointHoverRadius: 9,
    }));
  }, []);

  useEffect(() => {
    if (!ref.current) return;
    const ex = Chart.getChart(ref.current);
    if (ex) ex.destroy();
    new Chart(ref.current, {
      type: 'scatter',
      data: {datasets},
      plugins: [leaderLabelPlugin],
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {display: false},
          tooltip: {callbacks: {label: ctx => ctx.raw.label + '  ' + ctx.raw.x + 'B  ' + ctx.raw.y.toFixed(4)}}
        },
        layout: { padding: { top: 35, right: 45, bottom: 35, left: 45 } },
        scales: {
          x: {
            grid: {color:'#f0f0f0'},
            ticks: {font:{size:11}, color:'#888', callback: v => v + 'B'},
            title: {display:true, text:'Parametros (B)', font:{size:11}, color:'#888'}
          },
          y: {
            min: 0, max: 1,
            grid: {color:'#f0f0f0'},
            ticks: {font:{size:11}, color:'#888'},
            title: {display:true, text:'Hate F1', font:{size:11}, color:'#888'}
          }
        }
      }
    });
  }, []);

  return <div className="chart-box-scatter"><canvas ref={ref}/></div>;
}

function Graficos({ds}) {
  const isOv   = ds === '__overall__';
  const dsName = isOv ? 'todos los datasets' : (DATA.datasets_meta[ds]?.display || ds);
  return (
    <div className="section">
      <div className="section-header">
        <h2>Analisis grafico</h2>
        <span className="section-sub">{isOv ? 'Overall' : dsName}</span>
      </div>
      <div className="chart-label">
        Hate F1 {isOv ? 'medio sobre todos los datasets' : 'en ' + dsName}, ordenado de mayor a menor
      </div>
      <div className="chart-wrap">
        <HateF1Bar ds={ds}/>
      </div>
      <div className="chart-label">
        Hate F1 {isOv ? 'medio' : 'en ' + dsName} vs fecha de lanzamiento del modelo
      </div>
      <div className="chart-wrap">
        <ReleaseDateScatter ds={ds}/>
      </div>
      <div className="chart-label">
        Hate F1 {isOv ? 'medio' : 'en ' + dsName} vs numero de parametros del modelo
      </div>
      <div className="chart-wrap">
        <ParamsScatter ds={ds}/>
      </div>
    </div>
  );
}

function Datasets() {
  return (
    <div className="section">
      <div className="section-header">
        <h2>Datasets</h2>
        <span className="section-sub">{DATA.n_datasets} benchmarks en espanol</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>Dataset</th><th>Paper</th><th>Plataforma</th><th>Fuente</th>
            <th style={{textAlign:'right'}}>Total</th>
            <th style={{textAlign:'right'}}>Hate</th>
            <th style={{textAlign:'right'}}>No hate</th>
            <th>IAA</th><th>Tema</th><th>Lengua</th>
          </tr>
        </thead>
        <tbody>
          {DATA.datasets.map(d => {
            const m = DATA.datasets_meta[d] || {};
            return (
              <tr key={d}>
                <td><b>{m.display || d}</b></td>
                <td style={{color:'#888', fontSize:11}}>{m.paper}</td>
                <td>{m.platform}</td>
                <td style={{color:'#888'}}>{m.source}</td>
                <td style={{textAlign:'right'}}>{m.n_total ? m.n_total.toLocaleString() : '-'}</td>
                <td style={{textAlign:'right', color:'#ef4444'}}>{m.hate ? m.hate.toLocaleString() : '-'}</td>
                <td style={{textAlign:'right', color:'#10b981'}}>{m.no_hate ? m.no_hate.toLocaleString() : '-'}</td>
                <td style={{fontSize:11}}>{m.iaa || '-'}</td>
                <td style={{fontSize:11, color:'#888'}}>{m.topic}</td>
                <td><span className="tag">{m.lang}</span></td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function History() {
  const [showAll, setShowAll] = useState(false);
  const visible = showAll ? DATA.history : DATA.history.slice(0, 5);

  return (
    <div className="section">
      <div className="section-header">
        <h2>Historial de evaluaciones</h2>
        <span className="section-sub">{DATA.n_runs} runs totales</span>
      </div>
      <table>
        <thead>
          <tr>
            <th>Fecha</th><th>Modelo</th><th>Dataset</th>
            <th style={{textAlign:'right'}}>Macro-F1</th>
            <th style={{textAlign:'right'}}>Hate F1</th>
            <th style={{textAlign:'right'}}>Tiempo (s)</th>
          </tr>
        </thead>
        <tbody>
          {visible.map((r, i) => (
            <tr key={i}>
              <td style={{color:'#aaa'}}>{r.ts}</td>
              <td><b>{r.model}</b></td>
              <td>{DATA.datasets_meta[r.dataset] ? DATA.datasets_meta[r.dataset].display : r.dataset}</td>
              <td style={{textAlign:'right'}}><b>{r.f1 ? r.f1.toFixed(4) : '-'}</b></td>
              <td style={{textAlign:'right'}}>{r.hate_f1 ? r.hate_f1.toFixed(4) : '-'}</td>
              <td style={{textAlign:'right', color:'#aaa'}}>{r.elapsed != null ? r.elapsed : '-'}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {DATA.history.length > 5 && (
        <button
          onClick={() => setShowAll(v => !v)}
          style={{marginTop:10, fontSize:12, color:'#6366f1', background:'none', border:'none', cursor:'pointer', padding:'4px 0'}}>
          {showAll ? 'Mostrar menos' : 'Ver todos (' + DATA.history.length + ' runs)'}
        </button>
      )}
    </div>
  );
}


function MmHateF1Bar() {
  const ref = useRef(null);
  const rows = DATA.mm_rows || [];

  const chartData = useMemo(() => {
    const sorted = [...rows].sort((a, b) => (b.hate_f1||0) - (a.hate_f1||0));
    return {
      labels:   sorted.map(r => r.display_name || r.model),
      values:   sorted.map(r => r.hate_f1 || 0),
      colors:   sorted.map(r => col(r.family || '')),
      families: sorted.map(r => r.family || ''),
    };
  }, []);

  useEffect(() => {
    if (!ref.current) return;
    const ex = Chart.getChart(ref.current);
    if (ex) ex.destroy();
    const dataset = {
      data:            chartData.values,
      backgroundColor: chartData.colors,
      borderRadius:    4,
      borderSkipped:   false,
      _families:       chartData.families,
    };
    new Chart(ref.current, {
      type: 'bar',
      data: { labels: chartData.labels, datasets: [dataset] },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        layout: { padding: { top: 28 } },
        plugins: {
          legend: {display: false},
          tooltip: {callbacks: {label: ctx => ' Hate F1: ' + ctx.raw.toFixed(4)}}
        },
        scales: {
          x: { grid: {display: false}, ticks: {font:{size:10}, color:'#444', maxRotation:35, minRotation:25} },
          y: { min: 0, max: 1, grid: {color:'#f0f0f0'}, ticks: {font:{size:11}, color:'#888'} }
        }
      }
    });
  }, []);

  return <div className="chart-box"><canvas ref={ref}/></div>;
}

function MmParamsScatter() {
  const ref = useRef(null);
  const rows = DATA.mm_rows || [];

  const datasets = useMemo(() => {
    const byFamily = {};
    rows.forEach(r => {
      const f = r.family || 'Otro';
      if (!byFamily[f]) byFamily[f] = [];
      const p = r.params_exact || parseFloat((r.params||'0').replace('B',''));
      if (!p) return;
      byFamily[f].push({ x: p, y: r.macro_f1_binary || 0, label: r.display_name || r.model });
    });
    return Object.entries(byFamily).map(([f, pts]) => ({
      label: f, data: pts,
      backgroundColor: col(f) + 'cc', borderColor: col(f),
      pointRadius: 7, pointHoverRadius: 9,
    }));
  }, []);

  useEffect(() => {
    if (!ref.current) return;
    const ex = Chart.getChart(ref.current);
    if (ex) ex.destroy();
    new Chart(ref.current, {
      type: 'scatter',
      data: {datasets},
      plugins: [leaderLabelPlugin],
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          legend: {display: false},
          tooltip: {callbacks: {label: ctx => ctx.raw.label + '  ' + ctx.raw.x + 'B  Macro-F1 bin: ' + ctx.raw.y.toFixed(4)}}
        },
        layout: { padding: { top: 35, right: 45, bottom: 35, left: 45 } },
        scales: {
          x: {
            grid: {color:'#f0f0f0'},
            ticks: {font:{size:11}, color:'#888', callback: v => v + 'B'},
            title: {display:true, text:'Parametros (B)', font:{size:11}, color:'#888'}
          },
          y: {
            min: 0, max: 1, grid: {color:'#f0f0f0'},
            ticks: {font:{size:11}, color:'#888'},
            title: {display:true, text:'Macro-F1 binario', font:{size:11}, color:'#888'}
          }
        }
      }
    });
  }, []);

  return <div className="chart-box-scatter"><canvas ref={ref}/></div>;
}

function MmGraficos() {
  return (
    <div className="section">
      <div className="section-header">
        <h2>Analisis grafico</h2>
        <span className="section-sub">Multi3Hate — 300 memes ES</span>
      </div>
      <div className="chart-label">Hate F1 en Multi3Hate, ordenado de mayor a menor</div>
      <div className="chart-wrap"><MmHateF1Bar/></div>
      <div className="chart-label">Macro-F1 binario vs numero de parametros del modelo</div>
      <div className="chart-wrap"><MmParamsScatter/></div>
    </div>
  );
}

function MultimodalSection() {
  const rows = DATA.mm_rows || [];
  if (!rows.length) return null;

  const maxBin = Math.max(...rows.map(r => r.macro_f1_binary || 0));
  const maxHate = Math.max(...rows.map(r => r.hate_f1 || 0));

  return (
    <div id="multimodal">
      <div style={{margin:'3rem 0 1.5rem', paddingBottom:'1rem', borderBottom:'2px solid #e0e0e0'}}>
        <h2 style={{fontSize:20, fontWeight:700, letterSpacing:'-.3px'}}>
          🖼️enchmark de Imagen
        </h2>
        <div style={{fontSize:13, color:'#888', marginTop:4}}>
          {rows.length} modelos vision-lenguaje (VLMs) — Multi3Hate (Bui et al. 2024) — 300 memes, 5 idiomas
        </div>
      </div>
      <div className="info-box" style={{background:'#f0fdf4', borderColor:'#86efac', color:'#166534'}}>
        Evaluacion zero-shot de VLMs sobre <b>Multi3Hate</b> — dataset multilingue y multicultural
        de memes en espanol con anotadores de distintos paises. Gold labels: anotadores mexicanos.
        Los VLMs tienden a ser conservadores y clasificar mayoritariamente como no_hate,
        especialmente el hate implicito y culturalmente codificado.
      </div>
    <div className="section">
      <div className="section-header">
        <h2>Leaderboard Multimodal</h2>
        <span className="section-sub">{rows.length} modelos — dataset: Multi3Hate</span>
      </div>
      <table>
        <thead>
          <tr>
            <th style={{width:28}}>#</th>
            <th>Modelo</th>
            <th>Familia</th>
            <th>Params</th>
            <th>Backend</th>
            <th>Developer</th>
            <th>Macro-F1 bin ↓</th>
            <th>Hate F1</th>
            <th>Accuracy</th>
            <th>Cobertura</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => {
            const rk = i===0 ? 'r1' : i===1 ? 'r2' : i===2 ? 'r3' : 'rank';
            const pct = maxBin > 0 ? (r.macro_f1_binary || 0) / maxBin * 100 : 0;
            const hpct = maxHate > 0 ? (r.hate_f1 || 0) / maxHate * 100 : 0;
            const covColor = r.coverage >= 95 ? '#10b981' : r.coverage >= 80 ? '#f59e0b' : '#ef4444';
            const iconUrl = FAMILY_ICON[r.family];
            return (
              <tr key={r.model}>
                <td className={rk}>{i+1}</td>
                <td>
                  {iconUrl
                    ? <img src={iconUrl} width={16} height={16}
                        style={{marginRight:6, verticalAlign:'middle', borderRadius:2}}
                        onError={e => { e.target.style.display='none'; }}/>
                    : null}
                  <b>{r.display_name || r.model}</b>
                  <span style={{fontSize:10, color:'#aaa', marginLeft:6}}>{r.timestamp}</span>
                </td>
                <td style={{color: col(r.family), fontWeight:500}}>{r.family}</td>
                <td><span className="tag">{r.params}</span></td>
                <td style={{fontSize:11}}>
                  <span style={{
                    padding:'2px 7px', borderRadius:12, fontSize:10,
                    background: r.backend==='vllm' ? '#eff6ff' : '#f0fdf4',
                    color: r.backend==='vllm' ? '#1d4ed8' : '#166534'
                  }}>{r.backend}</span>
                </td>
                <td style={{color:'#888', fontSize:12}}>{r.developer}</td>
                <td>
                  <div className="bar-wrap">
                    <div className="bar-bg">
                      <div className="bar-fill" style={{width: pct+'%', background: col(r.family)}}/>
                    </div>
                    <span style={{minWidth:44, textAlign:'right', fontVariantNumeric:'tabular-nums'}}>
                      {r.macro_f1_binary != null ? r.macro_f1_binary.toFixed(4) : '-'}
                    </span>
                  </div>
                </td>
                <td>
                  <div className="bar-wrap">
                    <div className="bar-bg">
                      <div className="bar-fill" style={{width: hpct+'%', background:'#ef4444'}}/>
                    </div>
                    <span style={{minWidth:44, textAlign:'right', fontVariantNumeric:'tabular-nums'}}>
                      {r.hate_f1 != null ? r.hate_f1.toFixed(4) : '-'}
                    </span>
                  </div>
                </td>
                <td>{r.accuracy != null ? r.accuracy.toFixed(4) : '-'}</td>
                <td>
                  <span style={{color: covColor, fontWeight:500}}>
                    {r.coverage != null ? r.coverage.toFixed(1) + '%' : '-'}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <div style={{marginTop:8, fontSize:11, color:'#aaa'}}>
        * Macro-F1 binario calculado solo sobre predicciones hate/no_hate (excluye unclear).
        Hate F1 calculado sobre las 3 clases (hate, no_hate, unclear).
        Dataset: Multi3Hate — Bui et al. (2024) — 300 memes ES, gold=anotadores MX.
      </div>
    </div>
    <MmGraficos/>
    </div>
  );
}

function App() {
  const [ds, setDs] = useState('__overall__');
  const gen = DATA.generated ? DATA.generated.slice(0,16).replace('T',' ') + ' UTC' : '';
  return (
    <div>
      <a className="gh-btn" href="https://github.com/AritzUMA/hate-speech-bench-es" target="_blank" rel="noopener">
        <svg width="15" height="15" viewBox="0 0 16 16" fill="currentColor">
          <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
        </svg>
        AritzUMA/hate-speech-bench-es
      </a>
      <div className="wrap">
        <div className="header">
          <h1>
            <span className="live"/>
            Deteccion de Discurso de Odio en Espanol — Benchmark Continuo
          </h1>
          <div className="header-meta">
            Actualizado: {gen} - {DATA.n_models} modelos - {DATA.n_datasets} datasets - {DATA.n_runs} evaluaciones
          </div>
          <div className="header-desc">
            Benchmark continuo de modelos de lenguaje pequenos (SLMs, Small Language Models) para la deteccion
            de discurso de odio en espanol. Todos los modelos se evaluan en modo zero-shot mediante prompts JSON
            estructurados, sin fine-tuning ni adaptacion especifica a los datasets. Se utiliza temperatura=0
            para maximizar el determinismo de las respuestas. Las predicciones se clasifican en tres categorias:
            hate, no_hate y unclear (respuestas no parseables o ambiguas). Las metricas se reportan sobre las
            tres clases y tambien en version binaria excluyendo unclear.
          </div>
          <div className="header-method">
            Hardware: RTX 4070 Ti Super - 16GB VRAM - OLLAMA_NUM_PARALLEL=16
            &nbsp;·&nbsp;
            Evaluacion: zero-shot - temperatura=0 - sin reintentos
            &nbsp;·&nbsp;
            Etiquetas: hate - no_hate - unclear
          </div>
        </div>
        <div style={{margin:'0 0 1.5rem', paddingBottom:'1rem', borderBottom:'2px solid #e0e0e0'}}>
          <h2 style={{fontSize:20, fontWeight:700, letterSpacing:'-.3px'}}>
            Benchmark de Texto
          </h2>
          <div style={{fontSize:13, color:'#888', marginTop:4}}>
            {DATA.n_models} modelos — {DATA.n_datasets} datasets — {DATA.n_runs} evaluaciones
          </div>
        </div>
        <Leaderboard ds={ds} setDs={setDs}/>
        <Graficos ds={ds}/>
        <Datasets/>
        <History/>
        <MultimodalSection/>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App/>);
</script>
<script data-goatcounter="https://hate-speech-bench.goatcounter.com/count" async src="//gc.zgo.at/count.js"></script>
</body>
</html>"""


RUNS_DIR_MM = REPO_ROOT / "results" / "runs_multimodal"


def load_mm_runs():
    runs = []
    for f in sorted(RUNS_DIR_MM.glob("*.json")):
        if f.name.startswith("partial_"):
            continue
        try:
            r = json.loads(f.read_text(encoding="utf-8"))
            # Solo runs validos (>0 predicciones no-unclear)
            if r.get("n_unclear", 0) < r.get("n_instances", 1):
                runs.append(r)
        except Exception as e:
            print(f"  [warn] skip mm {f.name}: {e}")
    return runs


def build_mm_index(mm_runs, models_meta):
    best = {}
    for r in sorted(mm_runs, key=lambda x: x["timestamp"]):
        best[r["model"]] = r

    rows = []
    for model, r in best.items():
        m = r["metrics"]
        meta = models_meta.get(model, {})
        n_i = r.get("n_instances", 0)
        n_u = r.get("n_unclear", 0)
        rows.append({
            "model":           model,
            "display_name":    meta.get("display_name", model),
            "family":          meta.get("family", r.get("family", "")),
            "params":          meta.get("params", r.get("params", "")),
            "params_exact":    meta.get("params_exact"),
            "developer":       meta.get("developer", ""),
            "backend":         meta.get("backend", r.get("backend", "ollama")),
            "macro_f1":        round(m.get("macro_f1", 0), 4),
            "macro_f1_binary": round(m.get("macro_f1_binary", 0), 4),
            "hate_f1":         round(r["per_class"].get("hate", {}).get("f1-score", 0), 4),
            "accuracy":        round(m.get("accuracy", 0), 4),
            "coverage":        round((n_i - n_u) / n_i * 100, 1) if n_i else 0,
            "n_instances":     n_i,
            "n_unclear":       n_u,
            "timestamp":       r["timestamp"][:10],
        })

    rows.sort(key=lambda x: x["macro_f1_binary"], reverse=True)
    return rows


def main():
    DOCS_DIR.mkdir(exist_ok=True)
    runs = load_runs()
    if not runs:
        print("[gen] No hay runs — nada que generar")
        return
    models_meta, datasets_meta = load_registry()
    index = build_index(runs, models_meta, datasets_meta)

    # Multimodal
    mm_runs = load_mm_runs()
    mm_index = build_mm_index(mm_runs, models_meta)
    index["mm_rows"] = mm_index
    index["mm_n_models"] = len(mm_index)

    (DOCS_DIR / "runs_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False))
    print(f"[gen] {index['n_runs']} runs - {index['n_models']} modelos - {index['n_datasets']} datasets")
    print(f"[gen] multimodal: {len(mm_index)} modelos validos")
    html_path = DOCS_DIR / "index.html"
    html_path.write_text(generate_html(index), encoding="utf-8")
    print(f"[gen] index.html -> {html_path}")

if __name__ == "__main__":
    main()
