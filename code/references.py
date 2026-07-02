#!/usr/bin/env python3
"""
Non-deep references under the identical protocol:
  * multi-output Ridge (alpha chosen on validation)
  * XGBoost (ordinal-coded cats, like v5)
  * LightGBM (native categoricals)
  * CatBoost (native categoricals)            <-- NEW per review
  * non-learned baselines: training mean, per-line persistence,
    per-line 7-day rolling mean

Each learned reference returns test predictions (original units) for both
targets, plus validation DHU predictions for breach-alert cutoff tuning.
"""
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import config as C
from data import native_cat_frames


def _pack(pred_eff, pred_dhu, val_dhu=None, name=""):
    return dict(model=name, pred_eff=np.asarray(pred_eff, float),
                pred_dhu=np.asarray(pred_dhu, float),
                val_dhu=None if val_dhu is None else np.asarray(val_dhu, float))


# --------------------------------------------------------------------- ridge
def run_ridge(splits, alphas=C.RIDGE_ALPHAS):
    Xtr, Xvl, Xte = splits.train.Xord, splits.val.Xord, splits.test.Xord
    ytr, yvl = splits.train.ycols, splits.val.ycols
    tsc = splits.target_scaler
    best = (None, np.inf)
    for a in alphas:
        m = Ridge(alpha=a).fit(Xtr, ytr)
        v = mean_squared_error(yvl, m.predict(Xvl))
        if v < best[1]:
            best = (a, v)
    m = Ridge(alpha=best[0]).fit(Xtr, ytr)
    pred = tsc.inverse_transform(m.predict(Xte))
    vpred = tsc.inverse_transform(m.predict(Xvl))
    out = _pack(pred[:, 0], pred[:, 1], vpred[:, 1], name="Ridge")
    out["alpha"] = best[0]
    return out


# --------------------------------------------------------------------- xgboost
def run_xgboost(splits, seed=42, params=C.XGB_PARAMS):
    import xgboost as xgb
    Xtr, Xvl, Xte = splits.train.Xord, splits.val.Xord, splits.test.Xord
    yo_tr = splits.train.yorig; yo_vl = splits.val.yorig
    pe, pd_, vd = [], [], None
    preds = {}
    for j, task in enumerate(["eff", "dhu"]):
        m = xgb.XGBRegressor(random_state=seed, early_stopping_rounds=C.EARLY_STOP_ROUNDS,
                             eval_metric="rmse", **params)
        m.fit(Xtr, yo_tr[:, j], eval_set=[(Xvl, yo_vl[:, j])], verbose=False)
        preds[task] = m.predict(Xte)
        if task == "dhu":
            vd = m.predict(Xvl)
    return _pack(preds["eff"], preds["dhu"], vd, name="XGBoost")


# --------------------------------------------------------------------- lightgbm
def run_lightgbm(splits, seed=42, params=C.LGB_PARAMS):
    import lightgbm as lgb
    Ltr, Lvl, Lte, cats = native_cat_frames(splits)
    yo_tr = splits.train.yorig; yo_vl = splits.val.yorig
    preds = {}; vd = None
    for j, task in enumerate(["eff", "dhu"]):
        m = lgb.LGBMRegressor(random_state=seed, **params)
        m.fit(Ltr, yo_tr[:, j], eval_set=[(Lvl, yo_vl[:, j])],
              categorical_feature=cats,
              callbacks=[lgb.early_stopping(C.EARLY_STOP_ROUNDS, verbose=False)])
        preds[task] = m.predict(Lte)
        if task == "dhu":
            vd = m.predict(Lvl)
    return _pack(preds["eff"], preds["dhu"], vd, name="LightGBM")


# --------------------------------------------------------------------- catboost
def run_catboost(splits, seed=42, params=C.CAT_PARAMS):
    from catboost import CatBoostRegressor, Pool
    Ltr, Lvl, Lte, cats = native_cat_frames(splits)
    # CatBoost wants categorical columns as str, not pandas category dtype
    for fr in (Ltr, Lvl, Lte):
        for c in cats:
            fr[c] = fr[c].astype(str)
    cat_idx = [Ltr.columns.get_loc(c) for c in cats]
    yo_tr = splits.train.yorig; yo_vl = splits.val.yorig
    preds = {}; vd = None
    for j, task in enumerate(["eff", "dhu"]):
        m = CatBoostRegressor(random_seed=seed, early_stopping_rounds=C.EARLY_STOP_ROUNDS,
                              **params)
        m.fit(Pool(Ltr, yo_tr[:, j], cat_features=cat_idx),
              eval_set=Pool(Lvl, yo_vl[:, j], cat_features=cat_idx))
        preds[task] = m.predict(Lte)
        if task == "dhu":
            vd = m.predict(Lvl)
    return _pack(preds["eff"], preds["dhu"], vd, name="CatBoost")


# --------------------------------------------------------------- non-learned
def naive_baselines(splits):
    tr, te, vl = splits.train.part, splits.test.part, splits.val.part
    out = {}
    me, md = tr[C.TARGET_EFF].mean(), tr[C.TARGET_DHU].mean()
    n = len(te)
    out["TrainingMean"] = _pack(np.full(n, me), np.full(n, md), name="TrainingMean")
    out["Persistence"] = _pack(te["EffLag1"].values, te["DHULag1"].values,
                               vl["DHULag1"].values, name="Persistence")
    out["RollingMean7"] = _pack(te["EffRoll7"].values, te["DHURoll7"].values,
                                vl["DHURoll7"].values, name="RollingMean7")
    return out


def run_all_references(splits, seed=42, include=("ridge", "xgb", "lgbm", "cat")):
    out = {}
    if "ridge" in include: out["Ridge"] = run_ridge(splits)
    if "xgb" in include:   out["XGBoost"] = run_xgboost(splits, seed)
    if "lgbm" in include:  out["LightGBM"] = run_lightgbm(splits, seed)
    if "cat" in include:   out["CatBoost"] = run_catboost(splits, seed)
    out.update(naive_baselines(splits))
    return out
