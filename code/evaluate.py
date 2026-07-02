#!/usr/bin/env python3
"""
Evaluation utilities (rebuilt to the v5 protocol):
  * regression metrics (MAE, RMSE, R2 per task + average R2)
  * line-clustered AND i.i.d. bootstrap CIs for models and pairwise diffs
    (vectorised via per-cluster sufficient statistics -- exact pooled R2)
  * Holm-Bonferroni correction over a family of paired seed t-tests
  * validation-tuned breach-alert metrics (precision/recall/F1)
  * min-history bins
"""
import math
import numpy as np
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                             precision_score, recall_score, f1_score)
from scipy import stats


def summarize(yt, pe, pdh):
    pe = np.asarray(pe, float); pdh = np.asarray(pdh, float)
    return dict(
        eff_mae=mean_absolute_error(yt[:, 0], pe),
        eff_rmse=math.sqrt(mean_squared_error(yt[:, 0], pe)),
        eff_r2=r2_score(yt[:, 0], pe),
        dhu_mae=mean_absolute_error(yt[:, 1], pdh),
        dhu_rmse=math.sqrt(mean_squared_error(yt[:, 1], pdh)),
        dhu_r2=r2_score(yt[:, 1], pdh),
        avg_r2=(r2_score(yt[:, 0], pe) + r2_score(yt[:, 1], pdh)) / 2,
    )


# --------------------------------------------------------------- bootstrap
def _suff_stats(yte, models, cluster_ids):
    uniq, inv = np.unique(cluster_ids, return_inverse=True)
    k = len(uniq)
    cnt = np.bincount(inv, minlength=k).astype(float)
    Sy = np.stack([np.bincount(inv, weights=yte[:, j], minlength=k) for j in range(2)])
    Sy2 = np.stack([np.bincount(inv, weights=yte[:, j] ** 2, minlength=k) for j in range(2)])
    SSE = {}
    for m, (pe, pdh) in models.items():
        SSE[m] = np.stack([
            np.bincount(inv, weights=(yte[:, 0] - np.asarray(pe, float)) ** 2, minlength=k),
            np.bincount(inv, weights=(yte[:, 1] - np.asarray(pdh, float)) ** 2, minlength=k)])
    return k, cnt, Sy, Sy2, SSE


def bootstrap(yte, models, cluster_ids, pairs, B=2000, seed=7):
    """models: {name: (pred_eff, pred_dhu)}. Returns per-model CIs and pairwise
    difference CIs for avg/eff/dhu R2."""
    k, cnt, Sy, Sy2, SSE = _suff_stats(yte, models, cluster_ids)
    rg = np.random.default_rng(seed)
    M = np.stack([np.bincount(rg.integers(0, k, k), minlength=k) for _ in range(B)]).astype(float)
    N = M @ cnt
    r2s = {}
    for j in range(2):
        Sy_b = M @ Sy[j]
        SSt = M @ Sy2[j] - Sy_b ** 2 / N
        r2s[j] = {m: 1.0 - (M @ SSE[m][j]) / SSt for m in models}
    boot = {m: {"eff_r2": r2s[0][m], "dhu_r2": r2s[1][m],
                "avg_r2": (r2s[0][m] + r2s[1][m]) / 2} for m in models}

    def ci(a):
        a = np.sort(np.asarray(a)); kk = len(a)
        return [float(a[int(0.025 * kk)]), float(a[int(0.975 * kk)])]

    model_ci = {m: {key: ci(v) for key, v in d.items()} for m, d in boot.items()}
    diffs = {}
    for key in ["avg_r2", "eff_r2", "dhu_r2"]:
        for a, b in pairs:
            if a not in boot or b not in boot:
                continue
            d = np.asarray(boot[a][key]) - np.asarray(boot[b][key])
            c = ci(d)
            diffs[f"{a}-{b}_{key}"] = dict(mean=float(d.mean()), ci=c,
                                           excl0=bool(c[0] > 0 or c[1] < 0))
    return dict(model_ci=model_ci, diffs=diffs)


# --------------------------------------------------------------- Holm
def holm(pvalues):
    """Holm-Bonferroni. pvalues: dict name->p. Returns dict name->adjusted p."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adj = {}
    running = 0.0
    for i, (name, p) in enumerate(items):
        a = (m - i) * p
        running = max(running, a)
        adj[name] = min(1.0, running)
    return adj


def paired_seed_tests(per_seed, metric="avg_r2"):
    """per_seed: {model: [metric over seeds]}. Returns raw p for every unordered
    pair (paired t-test across seeds) ready to feed holm()."""
    names = list(per_seed)
    raw = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            xa, xb = np.asarray(per_seed[a]), np.asarray(per_seed[b])
            if len(xa) != len(xb) or len(xa) < 2:
                continue
            t, p = stats.ttest_rel(xa, xb)
            raw[f"{a}-{b}"] = float(p)
    return raw


# --------------------------------------------------------------- breach alert
def tuned_breach_alert(yte_dhu, yvl_dhu, test_preds, val_preds, thresholds):
    """Tune the decision cutoff per model on validation F1, freeze, apply to test."""
    out = {}
    for thr in thresholds:
        yv = (yvl_dhu > thr).astype(int)
        yt = (yte_dhu > thr).astype(int)
        out[str(thr)] = {}
        for m, vp in val_preds.items():
            best = (thr, -1.0)
            for c in np.arange(thr - 5, thr + 5.01, 0.25):
                f = f1_score(yv, (np.asarray(vp) > c).astype(int), zero_division=0)
                if f > best[1]:
                    best = (float(c), f)
            yh = (np.asarray(test_preds[m]) > best[0]).astype(int)
            out[str(thr)][m] = dict(
                cutoff=best[0],
                precision=float(precision_score(yt, yh, zero_division=0)),
                recall=float(recall_score(yt, yh, zero_division=0)),
                f1=float(f1_score(yt, yh, zero_division=0)))
    return out


def nominal_breach_alert(yte_dhu, test_preds, thresholds):
    out = {}
    for thr in thresholds:
        yt = (yte_dhu > thr).astype(int)
        out[str(thr)] = {}
        for m, tp in test_preds.items():
            yh = (np.asarray(tp) > thr).astype(int)
            out[str(thr)][m] = dict(
                precision=float(precision_score(yt, yh, zero_division=0)),
                recall=float(recall_score(yt, yh, zero_division=0)),
                f1=float(f1_score(yt, yh, zero_division=0)))
    return out


# --------------------------------------------------------------- history bins
def min_history_bins(df, test_part, yte, model_preds, line_col):
    hist_count = df.groupby(line_col).cumcount()
    te_hist = hist_count.loc[test_part.index].values
    rows = []
    for lo, hi in [(0, 4), (5, 9), (10, 19), (20, 59), (60, 10 ** 9)]:
        m = (te_hist >= lo) & (te_hist <= hi)
        if m.sum() < 10:
            continue
        row = dict(bin=f'{lo}-{hi if hi < 10**9 else "+"}', n=int(m.sum()))
        for name, (pe, pdh) in model_preds.items():
            row[f"{name}_avg_r2"] = summarize(yte[m], np.asarray(pe)[m],
                                              np.asarray(pdh)[m])["avg_r2"]
        rows.append(row)
    return rows
