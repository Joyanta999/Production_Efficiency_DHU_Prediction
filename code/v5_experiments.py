#!/usr/bin/env python3
"""
v5 audit-response experiments (FV2). Fixes the four substantive issues found
in the v4 review:

  A1  CLUSTER bootstrap by production line (B=2000) for all headline CIs and
      pairwise differences. The v4 row-level (i.i.d.) bootstrap ignored the
      clustering of test rows within lines; rows of one line share level and
      residual correlation, so row resampling understates uncertainty.
      The line-cluster bootstrap resamples whole lines with replacement.
      (Clustering by date is not viable: the test window holds only 10 dates.)
      The i.i.d. bootstrap is re-run alongside for transparency.
  A2  ROLLING-MEAN baseline: predict each target with its own 7-day shifted
      per-line rolling mean (EffRoll7/DHURoll7) -- literally one of the model
      inputs. Full metrics row + inclusion in both bootstraps, the tuned
      breach-alert evaluation, the rolling-origin folds, and min-history bins.
  A3  Anonymized robustness artifact: per-factory keys mapped to
      Factory A/B/C (results/robustness_v5_anon.json is shippable).
  A4  Corrected dataset facts: zero-DHU rows removed by the DHU floor after
      the efficiency filter (and the raw zero count), and the median
      within-line SD of ManPowerPresent.

Reads the canonical pipeline from ../../code/gbt_baseline.py (identical
protocol; torch-free fallback) and TabTransformer test predictions from
../../results/best_predictions.npz. Writes ../results/v5_results.json,
../results/baselines_v5.csv, ../results/robustness_v5_anon.json.
Nothing in ../../results/ is modified.
"""
import os, sys, json, math, copy
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ORIG_CODE = os.path.join(HERE, '..', '..', 'code')
ORIG_RES  = os.path.join(HERE, '..', '..', 'results')
OUT_RES   = os.path.join(HERE, '..', 'results')
os.makedirs(OUT_RES, exist_ok=True)
sys.path.insert(0, ORIG_CODE)

from gbt_baseline import CONFIG, load_dataframe, make_splits, PARAMS, SEEDS
from sklearn.linear_model import Ridge
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             precision_score, recall_score, f1_score)
import xgboost as xgb
import lightgbm as lgb

OUT = {}
rng = np.random.default_rng(7)

# ---------------- base data, identical protocol ----------------
df = load_dataframe(CONFIG)
splits, feats, ts, parts = make_splits(df, CONFIG)
Xtr, ytr, _ = splits['train']; Xvl, yvl, _ = splits['val']; Xte, yte, _ = splits['test']
tr, vl, te = parts['train'], parts['val'], parts['test']
yte_o = np.column_stack([te['AchievedEfficiency'].values, te['dhu'].values])
ytr_o = ts.inverse_transform(ytr); yvl_o = ts.inverse_transform(yvl)
n = len(yte_o)
lines_te = te['WorkspaceLineId'].values

def summarize(yt, pe, pdh):
    pe, pdh = np.asarray(pe, float), np.asarray(pdh, float)
    return dict(eff_mae=mean_absolute_error(yt[:,0], pe),
                eff_rmse=math.sqrt(mean_squared_error(yt[:,0], pe)),
                eff_r2=r2_score(yt[:,0], pe),
                dhu_mae=mean_absolute_error(yt[:,1], pdh),
                dhu_rmse=math.sqrt(mean_squared_error(yt[:,1], pdh)),
                dhu_r2=r2_score(yt[:,1], pdh),
                avg_r2=(r2_score(yt[:,0], pe)+r2_score(yt[:,1], pdh))/2)

# ---------------- model predictions on the fixed test set ----------------
# XGBoost seed 42 (deterministic given seed), mirrors v4_experiments.py
px, px_val = {}, {}
for j, task in enumerate(['eff','dhu']):
    m = xgb.XGBRegressor(random_state=42, **PARAMS)
    m.fit(Xtr, ytr_o[:,j], eval_set=[(Xvl, yvl_o[:,j])], verbose=False)
    px[task] = m.predict(Xte); px_val[task] = m.predict(Xvl)

# LightGBM seed 42, native categoricals, mirrors v4_experiments.py
NUMF = [c for c in feats if not c.endswith('_enc')]
CATS = ['WorkspaceFactoryName', 'WorkspaceBuildingName', 'BuyerName']
def lgb_frame(part):
    X = part[NUMF].copy().fillna(tr[NUMF].median())
    for c in CATS:
        X[c] = pd.Categorical(part[c], categories=sorted(tr[c].unique()))
    return X
Ltr, Lvl, Lte = lgb_frame(tr), lgb_frame(vl), lgb_frame(te)
pl = {}
for j, task in enumerate(['eff','dhu']):
    m = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.05, num_leaves=63,
                          subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          reg_lambda=1.0, random_state=42, verbose=-1)
    m.fit(Ltr, ytr_o[:,j], eval_set=[(Lvl, yvl_o[:,j])],
          callbacks=[lgb.early_stopping(50, verbose=False)],
          categorical_feature=CATS)
    pl[task] = m.predict(Lte)

# Ridge (deterministic; alpha=100 was chosen on validation in v3)
ridge = Ridge(alpha=100).fit(Xtr, ytr)
pr   = ts.inverse_transform(ridge.predict(Xte))
pr_v = ts.inverse_transform(ridge.predict(Xvl))

# TabTransformer seed 42 predictions (canonical artifact)
tt = np.load(os.path.join(ORIG_RES, 'best_predictions.npz'))
assert np.allclose(tt['true_eff'], yte_o[:,0], atol=1e-3), 'test set mismatch'

# Non-learned forecasts (lag/roll columns are train-median-filled by make_splits)
pers_e, pers_d = te['EffLag1'].values, te['DHULag1'].values
roll_e, roll_d = te['EffRoll7'].values, te['DHURoll7'].values

models_te = {
 'lgbm':        (pl['eff'], pl['dhu']),
 'xgb':         (px['eff'], px['dhu']),
 'ridge':       (pr[:,0], pr[:,1]),
 'tabt':        (tt['pred_eff'], tt['pred_dhu']),
 'rollmean':    (roll_e, roll_d),
 'persistence': (pers_e, pers_d),
}

# ---------------- A2: rolling-mean baseline metrics (Table II row) ----------
base_rows = []
trm_e, trm_d = tr['AchievedEfficiency'].mean(), tr['dhu'].mean()
base_rows.append(dict(model='TrainingMean',
                      **summarize(yte_o, np.full(n, trm_e), np.full(n, trm_d))))
base_rows.append(dict(model='Persistence',  **summarize(yte_o, pers_e, pers_d)))
base_rows.append(dict(model='RollingMean7', **summarize(yte_o, roll_e, roll_d)))
base_rows.append(dict(model='Ridge_a100',   **summarize(yte_o, pr[:,0], pr[:,1])))
bdf = pd.DataFrame(base_rows)
bdf.to_csv(os.path.join(OUT_RES, 'baselines_v5.csv'), index=False)
OUT['baselines'] = json.loads(bdf.to_json(orient='records'))
print(bdf[['model','eff_r2','dhu_r2','avg_r2']].round(3).to_string(index=False))

# ---------------- A1: cluster bootstrap by line + i.i.d. rerun --------------
# Vectorized via per-cluster sufficient statistics: for a resample that picks
# clusters c with multiplicity m_c,
#   R^2 = 1 - sum_c m_c*SSE_c / (sum_c m_c*Sy2_c - (sum_c m_c*Sy_c)^2 / N)
# which is exact for the pooled R^2 of the concatenated resample.
B = 2000
uniq_lines = np.unique(lines_te)
L = len(uniq_lines)
PAIRS = [('lgbm','xgb'), ('xgb','tabt'), ('ridge','tabt'), ('ridge','xgb'),
         ('xgb','persistence'), ('tabt','rollmean'), ('rollmean','persistence'),
         ('xgb','rollmean')]
MODELS = list(models_te)

def suff_stats(cluster_ids):
    """Per-cluster n, sum(y), sum(y^2) per task and SSE per model/task."""
    uniq, inv = np.unique(cluster_ids, return_inverse=True)
    k = len(uniq)
    cnt = np.bincount(inv, minlength=k).astype(float)
    Sy  = np.stack([np.bincount(inv, weights=yte_o[:,j], minlength=k) for j in range(2)])
    Sy2 = np.stack([np.bincount(inv, weights=yte_o[:,j]**2, minlength=k) for j in range(2)])
    SSE = {}
    for m, (pe, pdh) in models_te.items():
        SSE[m] = np.stack([
            np.bincount(inv, weights=(yte_o[:,0]-np.asarray(pe, float))**2, minlength=k),
            np.bincount(inv, weights=(yte_o[:,1]-np.asarray(pdh, float))**2, minlength=k)])
    return k, cnt, Sy, Sy2, SSE

def run_bootstrap(cluster_ids, tag, seed=7):
    k, cnt, Sy, Sy2, SSE = suff_stats(cluster_ids)
    rg = np.random.default_rng(seed)
    # multiplicity matrix: B x k
    M = np.stack([np.bincount(rg.integers(0, k, k), minlength=k) for _ in range(B)]).astype(float)
    N   = M @ cnt                       # B
    r2s = {}
    for j in range(2):
        Sy_b  = M @ Sy[j]
        SSt_b = M @ Sy2[j] - Sy_b**2 / N
        r2s[j] = {m: 1.0 - (M @ SSE[m][j]) / SSt_b for m in MODELS}
    boot = {m: {'eff_r2': r2s[0][m], 'dhu_r2': r2s[1][m],
                'avg_r2': (r2s[0][m] + r2s[1][m]) / 2} for m in MODELS}
    def ci(a):
        a = np.sort(np.asarray(a)); kk = len(a)
        return [float(a[int(0.025*kk)]), float(a[int(0.975*kk)])]
    OUT[f'bootstrap_ci_{tag}'] = {m: {kk: ci(v) for kk, v in d.items()}
                                  for m, d in boot.items()}
    diffs = {}
    for kk in ['avg_r2','eff_r2','dhu_r2']:
        for a, b_ in PAIRS:
            d = np.asarray(boot[a][kk]) - np.asarray(boot[b_][kk])
            c = ci(d)
            diffs[f'{a}-{b_}_{kk}'] = dict(mean=float(d.mean()), ci=c,
                                           excl0=bool(c[0] > 0 or c[1] < 0))
    OUT[f'bootstrap_diffs_{tag}'] = diffs
    print(f'--- {tag} bootstrap, avg_r2 differences ---')
    for a, b_ in PAIRS:
        v = diffs[f'{a}-{b_}_avg_r2']
        print(f"  {a}-{b_}: {v['mean']:+.3f}  CI [{v['ci'][0]:+.3f}, {v['ci'][1]:+.3f}]"
              f"  excl0={v['excl0']}")

run_bootstrap(np.arange(n), 'iid')          # every row its own cluster
run_bootstrap(lines_te, 'cluster_line')      # resample whole lines
OUT['n_test_lines'] = int(L)
OUT['n_test_dates'] = int(te['Date'].nunique())

# ---------------- A2 cont.: rolling mean in tuned breach alerts -------------
yvl_dhu = vl['dhu'].values
val_preds = {'xgb': px_val['dhu'], 'ridge': pr_v[:,1],
             'persistence': vl['DHULag1'].values, 'rollmean': vl['DHURoll7'].values}
test_preds = {'xgb': px['dhu'], 'ridge': pr[:,1],
              'persistence': pers_d, 'rollmean': roll_d}
tuned = {}
for thr in [10.0, 15.0]:
    yv = (yvl_dhu > thr).astype(int); yt_ = (yte_o[:,1] > thr).astype(int)
    tuned[str(thr)] = {}
    for mname, vp in val_preds.items():
        best = (thr, -1)
        for c in np.arange(thr-5, thr+5.01, 0.25):
            f = f1_score(yv, (np.asarray(vp) > c).astype(int), zero_division=0)
            if f > best[1]: best = (float(c), f)
        yh = (np.asarray(test_preds[mname]) > best[0]).astype(int)
        tuned[str(thr)][mname] = dict(cutoff=best[0],
            precision=float(precision_score(yt_, yh, zero_division=0)),
            recall=float(recall_score(yt_, yh, zero_division=0)),
            f1=float(f1_score(yt_, yh, zero_division=0)))
    print(thr, {m: round(d['f1'],3) for m, d in tuned[str(thr)].items()})
OUT['tuned_breach_v5'] = tuned
# nominal-threshold row for the rolling mean (completes paper Table V)
nominal = {}
for thr in [10.0, 15.0]:
    yt_ = (yte_o[:,1] > thr).astype(int)
    yh = (roll_d > thr).astype(int)
    nominal[str(thr)] = dict(precision=float(precision_score(yt_, yh, zero_division=0)),
                             recall=float(recall_score(yt_, yh, zero_division=0)),
                             f1=float(f1_score(yt_, yh, zero_division=0)))
OUT['nominal_breach_rollmean'] = nominal

# ---------------- A2 cont.: rolling mean in min-history bins ----------------
hist_count = df.groupby('WorkspaceLineId').cumcount()
te_hist = hist_count.loc[te.index].values
mh = []
for lo, hi in [(0,4),(5,9),(10,19),(20,59),(60,10**9)]:
    m_ = (te_hist >= lo) & (te_hist <= hi)
    if m_.sum() < 10: continue
    mh.append(dict(bin=f'{lo}-{hi if hi<10**9 else "+"}', n=int(m_.sum()),
                   xgb_avg_r2=summarize(yte_o[m_], px['eff'][m_], px['dhu'][m_])['avg_r2'],
                   pers_avg_r2=summarize(yte_o[m_], pers_e[m_], pers_d[m_])['avg_r2'],
                   roll_avg_r2=summarize(yte_o[m_], roll_e[m_], roll_d[m_])['avg_r2']))
OUT['min_history_v5'] = mh
for r_ in mh: print(r_)

# ---------------- A2 cont.: rolling mean in rolling-origin folds ------------
nfull = len(df); folds = []
for k in range(5):
    tr_end = int(nfull * (0.55 + 0.07*k)); te_end = min(int(nfull * (0.55 + 0.07*(k+1))), nfull)
    sub = df.iloc[:te_end]
    nk = len(sub); trk = int(tr_end*0.85); vlk = tr_end
    cfg2 = dict(CONFIG, train_frac=trk/nk, val_frac=(vlk-trk)/nk)
    spk, ftk, tsk, ptk = make_splits(sub, cfg2)
    tk = ptk['test']
    ytk = np.column_stack([tk['AchievedEfficiency'].values, tk['dhu'].values])
    rk = summarize(ytk, tk['EffRoll7'].values, tk['DHURoll7'].values)
    pk = summarize(ytk, tk['EffLag1'].values, tk['DHULag1'].values)
    folds.append(dict(fold=k, test_start=str(tk['Date'].min().date()),
                      test_end=str(tk['Date'].max().date()), n_test=len(tk),
                      roll_avg_r2=rk['avg_r2'], pers_avg_r2=pk['avg_r2']))
OUT['rolling_origin_rollmean'] = folds

# ---------------- A4: corrected dataset facts --------------------------------
raw = pd.read_csv(CONFIG['data_path'], encoding='utf-8-sig')
for c in ['AchievedEfficiency','dhu','ManPowerPresent']:
    raw[c] = pd.to_numeric(raw[c], errors='coerce')
raw = raw[raw['WorkspaceFactoryName'].str.contains('Ltd', na=False)]
eff_ok = raw[raw['AchievedEfficiency'].between(CONFIG['eff_min'], CONFIG['eff_max'])]
OUT['facts'] = dict(
    raw_records=int(len(raw)),
    zero_dhu_raw=int((raw['dhu'] == 0).sum()),
    zero_dhu_removed_by_dhu_floor=int((eff_ok['dhu'] == 0).sum()),
    sub_floor_nonzero_dhu=int(((eff_ok['dhu'] > 0) & (eff_ok['dhu'] < CONFIG['dhu_min'])).sum()),
    dhu_above_cap=int((eff_ok['dhu'] > CONFIG['dhu_max']).sum()),
    manpower_median_within_line_sd=float(
        raw.groupby('WorkspaceLineId')['ManPowerPresent'].std().median()),
)
print('facts:', OUT['facts'])

# ---------------- A3: anonymized robustness artifact -------------------------
key = json.load(open(os.path.join(ORIG_RES, 'anonymization_key.json')))
rob = json.load(open(os.path.join(ORIG_RES, 'robustness.json')))
rob = copy.deepcopy(rob)
rob['per_factory'] = {key['factories'].get(k, k): v
                      for k, v in rob['per_factory'].items()}
blob = json.dumps(rob)
for real in list(key['factories']) + list(key['buyers']):
    assert real not in blob, f'real name {real!r} survived anonymization'
json.dump(rob, open(os.path.join(OUT_RES, 'robustness_v5_anon.json'), 'w'), indent=1)
print('wrote robustness_v5_anon.json (clean)')

# ---------------- consistent-rounding ablation artifact ----------------------
abl = pd.read_csv(os.path.join(ORIG_RES, 'ablation_single_task.csv'))
ps_ = pd.read_csv(os.path.join(ORIG_RES, 'results_per_seed.csv'))
lines = ['| Architecture | Task | Single-task R2 | Multi-task R2 |', '|---|---|---|---|']
for arch in ['MLP','DeepMLP','TabTransformer','CNN1D','BiLSTM']:
    for task, col in [('eff','eff_r2'), ('dhu','dhu_r2')]:
        st = abl[(abl.model == arch) & (abl.task == task)]['r2']
        mt = ps_[ps_.model == arch][col]
        if len(st):
            lines.append(f'| {arch} | {task} | {st.mean():.3f} ± {st.std(ddof=1):.3f} '
                         f'| {mt.mean():.3f} ± {mt.std(ddof=1):.3f} |')
open(os.path.join(OUT_RES, 'ablation_table4_v5.md'), 'w').write(
    '# Table IV (v5) — consistent 3-dp rounding, same seeds/ddof as Table I\n\n'
    + '\n'.join(lines) + '\n')

json.dump(OUT, open(os.path.join(OUT_RES, 'v5_results.json'), 'w'), indent=1, default=str)
print('wrote v5_results.json')
