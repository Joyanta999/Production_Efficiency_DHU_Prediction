#!/usr/bin/env python3
"""
Rolling-origin evaluation for EVERY main model (v5 ran it for XGBoost only).

Five expanding-window folds: the train+val origin marches forward and each fold
tests on the next contiguous block of records. For every fold we retrain and
score all requested models -- trees, ridge, the naive baselines, and the deep
models -- so the headline ranking can be checked across windows, not just one
slice. `train` (and torch) is imported lazily so the tree-only path runs
without torch installed.
"""
import numpy as np

import config as C
from data import make_splits
from references import (run_ridge, run_xgboost, run_lightgbm, run_catboost,
                        naive_baselines)
from evaluate import summarize


def _splits_for_fold(df, k, n_folds=C.ROLLING_FOLDS):
    n = len(df)
    start = 0.55
    tr_end = int(n * (start + (0.45 / n_folds) * k))
    te_end = min(int(n * (start + (0.45 / n_folds) * (k + 1))), n)
    sub = df.iloc[:te_end].reset_index(drop=True)
    nk = len(sub)
    n_tr = int(tr_end * 0.85)
    cfg2 = dict(C.CONFIG, train_frac=n_tr / nk, val_frac=(tr_end - n_tr) / nk)
    return make_splits(sub, cfg2)


def rolling_origin(df, deep_models=("TabTransformer", "FTTransformer"),
                   tree_models=("xgb", "lgbm", "cat", "ridge"),
                   n_folds=C.ROLLING_FOLDS, dl_seed=42, hp_by_model=None):
    hp_by_model = hp_by_model or {}
    if deep_models:
        from train import train_model
    runners = dict(ridge=run_ridge, xgb=run_xgboost,
                   lgbm=run_lightgbm, cat=run_catboost)
    folds = []
    for k in range(n_folds):
        sp = _splits_for_fold(df, k, n_folds)
        te = sp.test.part
        yte = sp.test.yorig
        rec = dict(fold=k, n_test=len(te),
                   test_start=str(te[C.DATE_COL].min().date()),
                   test_end=str(te[C.DATE_COL].max().date()),
                   models={})

        for name, ref in naive_baselines(sp).items():
            rec["models"][name] = summarize(yte, ref["pred_eff"], ref["pred_dhu"])["avg_r2"]

        for key in tree_models:
            try:
                ref = runners[key](sp) if key == "ridge" else runners[key](sp, dl_seed)
                rec["models"][ref["model"]] = summarize(yte, ref["pred_eff"], ref["pred_dhu"])["avg_r2"]
            except Exception as e:
                rec["models"][key] = "ERROR: " + str(e)

        for name in deep_models:
            try:
                out = train_model(name, sp, dl_seed, hp=hp_by_model.get(name))
                rec["models"][name] = summarize(yte, out["pred_eff"], out["pred_dhu"])["avg_r2"]
            except Exception as e:
                rec["models"][name] = "ERROR: " + str(e)
        folds.append(rec)
    return folds
