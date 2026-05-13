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
        results[model][dataset] = {
            "macro_f1":        m.get("macro_f1", 0),
            "macro_f1_binary": m.get("macro_f1_binary", m.get("macro_f1", 0)),
            "precision":       m.get("precision", 0),
            "recall":          m.get("recall", 0),
            "accuracy":        m.get("accuracy", 0),
            "hate_f1":         r["per_class"].get("hate", {}).get("f1-score", 0),
            "n_instances":     r.get("n_instances", 0),
            "n_unclear":       r.get("n_unclear", r.get("n_sin_pred", 0)),
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
.chart-box{position:relative;height:180px}
.chart-box-scatter{position:relative;height:200px}
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
  'LLaMA 3.1':   CDN+'meta.png',
  'LLaMA':       CDN+'meta.png',
  'Gemma 2':     CDN+'gemma.png',
  'Gemma 3':     CDN+'gemma.png',
  'Gemma 4':     CDN+'gemma.png',
  'Qwen 2.5':    CDN+'qwen.png',
  'Qwen 3':      CDN+'qwen.png',
  'Qwen 3.5':    CDN+'qwen.png',
  'Mistral':     CDN+'mistral.png',
  'DeepSeek R1': CDN+'deepseek.png',
  'Phi 3':       CDN+'microsoft.png',
  'Phi 3.5':     CDN+'microsoft.png',
  'Yi':          CDN+'yi.png',
  'StableLM 2':  CDN+'stability-ai.png',
  'GPT-OSS':     CDN+'openai.png',
  'Salamandra':  LOGOS+'salamandra.png',
  'OpenEuroLLM': LOGOS+'openeurollm.png',
  'Moondream':   LOGOS+'moondream.png',
};

const COLORS = {
  'LLaMA 3.1':'#3b82f6','LLaMA':'#3b82f6',
  'Gemma 2':'#10b981','Gemma 3':'#059669','Gemma 4':'#047857',
  'Qwen 2.5':'#f59e0b','Qwen 3':'#d97706','Qwen 3.5':'#b45309',
  'Mistral':'#8b5cf6','DeepSeek R1':'#ec4899',
  'Salamandra':'#f97316','GPT-OSS':'#6b7280',
  'Yi':'#6366f1','Phi 3.5':'#14b8a6','Phi 3':'#0d9488',
  'StableLM 2':'#9ca3af','OpenEuroLLM':'#84cc16',
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

function Leaderboard() {
  const [ds, setDs]           = useState('__overall__');
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
            {isOv && <th>Cobertura</th>}
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
                {isOv && (
                  <td>
                    <span className="cov-dot" style={{background: covColor}}/>
                    {cov != null ? cov + '%' : '-'}
                  </td>
                )}
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function HateF1Bar() {
  const ref = useRef(null);

  const chartData = useMemo(() => {
    const models = DATA.models.filter(m => DATA.overall[m] && DATA.overall[m].hate_f1 != null);
    const sorted = [...models].sort((a, b) =>
      (DATA.overall[b].hate_f1 || 0) - (DATA.overall[a].hate_f1 || 0)
    );
    return {
      labels:   sorted.map(m => m),
      values:   sorted.map(m => DATA.overall[m].hate_f1 || 0),
      colors:   sorted.map(m => col(DATA.models_meta[m] ? DATA.models_meta[m].family : '')),
      families: sorted.map(m => DATA.models_meta[m] ? DATA.models_meta[m].family : ''),
    };
  }, []);

  useEffect(() => {
    if (!ref.current) return;
    const ex = Chart.getChart(ref.current);
    if (ex) ex.destroy();
    const ds = {
      data:            chartData.values,
      backgroundColor: chartData.colors,
      borderRadius:    4,
      borderSkipped:   false,
      _families:       chartData.families,
    };
    new Chart(ref.current, {
      type: 'bar',
      data: { labels: chartData.labels, datasets: [ds] },
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
  }, []);

  return <div className="chart-box"><canvas ref={ref}/></div>;
}

// Plugin de etiquetas con lineas conectoras y deteccion de colisiones
const leaderLabelPlugin = {
  id: 'leaderLabel',
  afterDraw(chart) {
    const ctx   = chart.ctx;
    const meta0 = chart.getDatasetMeta(0);
    if (!meta0) return;

    // Recopilar todos los puntos con etiqueta
    const items = [];
    chart.data.datasets.forEach((ds, di) => {
      const meta = chart.getDatasetMeta(di);
      ds.data.forEach((pt, pi) => {
        const el = meta.data[pi];
        if (!el) return;
        items.push({
          px: el.x,
          py: el.y,
          label: pt.label || '',
          color: ds.borderColor || '#555',
        });
      });
    });

    const FONT_SIZE = 10;
    const PAD      = 3;
    const LINE_LEN = 14;
    ctx.font        = `${FONT_SIZE}px system-ui,sans-serif`;
    ctx.textBaseline = 'middle';

    // Asignar posicion inicial alternando arriba/abajo
    const placed = items.map((item, i) => {
      const w   = ctx.measureText(item.label).width + PAD * 2;
      const h   = FONT_SIZE + PAD * 2;
      const dir = i % 2 === 0 ? -1 : 1;
      return {
        ...item,
        lx: item.px - w / 2,
        ly: item.py + dir * (LINE_LEN + h / 2),
        w, h,
      };
    });

    // Iteraciones de separacion de colisiones
    for (let iter = 0; iter < 30; iter++) {
      for (let a = 0; a < placed.length; a++) {
        for (let b = a + 1; b < placed.length; b++) {
          const A = placed[a], B = placed[b];
          const ox = Math.abs(A.lx + A.w/2 - (B.lx + B.w/2)) - (A.w + B.w) / 2;
          const oy = Math.abs(A.ly - B.ly) - (A.h + B.h) / 2 - 2;
          if (ox < 0 && oy < 0) {
            const push = Math.min(-oy / 2 + 2, 8);
            if (A.ly < B.ly) { A.ly -= push; B.ly += push; }
            else              { A.ly += push; B.ly -= push; }
          }
        }
      }
    }

    // Dibujar lineas y etiquetas
    placed.forEach(item => {
      const tx = item.lx;
      const ty = item.ly;
      const cx = item.px;
      const cy = item.py;

      // Linea conectora
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(tx + item.w / 2, ty);
      ctx.strokeStyle = item.color + '88';
      ctx.lineWidth   = 0.8;
      ctx.stroke();

      // Fondo semitransparente
      ctx.fillStyle = 'rgba(255,255,255,0.85)';
      ctx.fillRect(tx, ty - item.h / 2, item.w, item.h);

      // Texto
      ctx.fillStyle   = item.color;
      ctx.textAlign   = 'center';
      ctx.fillText(item.label, tx + item.w / 2, ty);
    });
    ctx.textAlign = 'left';
  }
};

function ReleaseDateScatter() {
  const ref = useRef(null);

  const datasets = useMemo(() => {
    const models = DATA.models.filter(m => DATA.overall[m]);
    const byFamily = {};
    models.forEach(m => {
      const meta = DATA.models_meta[m] || {};
      const f    = meta.family || 'Otro';
      if (!byFamily[f]) byFamily[f] = [];
      const parts = (meta.release_date || '').split('-').map(Number);
      const y = parts[0], mo = parts[1];
      if (!y) return;
      byFamily[f].push({ x: y + (mo-1)/12, y: DATA.overall[m].hate_f1 || 0, label: (DATA.models_meta[m] ? DATA.models_meta[m].display_name : m), mo, yr: y });
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
            return ctx.raw.label + '  (' + mo + ' ' + ctx.raw.yr + ')  ' + ctx.raw.y.toFixed(4);
          }}}
        },
        layout: { padding: { top: 20, right: 20, bottom: 20, left: 20 } },
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

function ParamsScatter() {
  const ref = useRef(null);

  const datasets = useMemo(() => {
    const models = DATA.models.filter(m => DATA.overall[m]);
    const byFamily = {};
    models.forEach(m => {
      const meta = DATA.models_meta[m] || {};
      const f    = meta.family || 'Otro';
      if (!byFamily[f]) byFamily[f] = [];
      const p = parseFloat((meta.params || '0').replace('B',''));
      if (!p) return;
      byFamily[f].push({ x: p, y: DATA.overall[m].hate_f1 || 0, label: (DATA.models_meta[m] ? DATA.models_meta[m].display_name : m) });
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
        layout: { padding: { top: 20, right: 20, bottom: 20, left: 20 } },
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

function Graficos() {
  return (
    <div className="section">
      <div className="section-header">
        <h2>Analisis grafico</h2>
      </div>
      <div className="chart-label">
        Hate F1 medio sobre todos los datasets evaluados, ordenado de mayor a menor
      </div>
      <div className="chart-wrap">
        <HateF1Bar/>
      </div>
      <div className="chart-label">
        Hate F1 medio vs fecha de lanzamiento del modelo — etiquetado por modelo
      </div>
      <div className="chart-wrap">
        <ReleaseDateScatter/>
      </div>
      <div className="chart-label">
        Hate F1 medio vs numero de parametros del modelo
      </div>
      <div className="chart-wrap">
        <ParamsScatter/>
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

function App() {
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
        <Leaderboard/>
        <Graficos/>
        <Datasets/>
        <History/>
      </div>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById('root')).render(<App/>);
</script>
</body>
</html>"""


def main():
    DOCS_DIR.mkdir(exist_ok=True)
    runs = load_runs()
    if not runs:
        print("[gen] No hay runs — nada que generar")
        return
    models_meta, datasets_meta = load_registry()
    index = build_index(runs, models_meta, datasets_meta)
    (DOCS_DIR / "runs_index.json").write_text(
        json.dumps(index, indent=2, ensure_ascii=False))
    print(f"[gen] {index['n_runs']} runs - {index['n_models']} modelos - {index['n_datasets']} datasets")
    html_path = DOCS_DIR / "index.html"
    html_path.write_text(generate_html(index), encoding="utf-8")
    print(f"[gen] index.html -> {html_path}")

if __name__ == "__main__":
    main()
