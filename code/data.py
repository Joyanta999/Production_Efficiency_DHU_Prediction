#!/usr/bin/env python3
"""
Leakage-audited data pipeline (rebuilt from scratch to the v5 protocol).

Pipeline:
  1. load raw export, coerce numerics, keep "Ltd" factories
  2. clean: efficiency in [eff_min, eff_max], dhu in [dhu_min, dhu_max]
  3. sort chronologically, tie-break by line id (deterministic ordering)
  4. engineer pre-shift features INCLUDING per-line history computed over
     RETAINED records (after cleaning), shifted by one so only strictly past
     values enter -- EffLag1/DHULag1 and 7-record rolling means EffRoll7/DHURoll7
  5. chronological 70/15/15 split
  6. fit ALL preprocessing on TRAIN ONLY:
        - median imputation (train medians)
        - RobustScaler on numeric features
        - StandardScaler on the two targets (metrics inverted afterwards)
        - categorical vocabularies (unseen -> reserved index 0)

Outputs a Splits object exposing, for each of train/val/test:
  Xnum   : scaled numeric matrix            (n, 19)        -> all models
  Xcat   : integer category indices         (n, 3)         -> neural embeddings
  Xord   : numeric + ordinal-coded cats     (n, 22)        -> ridge / xgboost
  ycols  : standardized targets             (n, 2)
  yorig  : original-scale targets           (n, 2)
  part   : the raw DataFrame slice (for line ids, dates, history, native cats)
Plus:
  cat_cardinalities : list[int] (incl. reserved index) for embedding tables
  target_scaler     : to invert predictions
  history columns EffLag1/DHULag1/EffRoll7/DHURoll7 live in `part` for the
  persistence and rolling-mean baselines.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import RobustScaler, StandardScaler

import config as C


# --------------------------------------------------------------------------- io
def load_raw(cfg=C.CONFIG):
    raw = pd.read_csv(cfg["data_path"], encoding="utf-8-sig")
    # coerce the fields we rely on numerically
    for col in [C.TARGET_EFF, C.TARGET_DHU] + C.PLAN_NUM_COLS:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")
    if cfg.get("factory_filter"):
        raw = raw[raw["WorkspaceFactoryName"].astype(str)
                  .str.contains(cfg["factory_filter"], na=False)]
    raw[C.DATE_COL] = pd.to_datetime(raw[C.DATE_COL], errors="coerce")
    return raw.reset_index(drop=True)


def clean(raw, cfg=C.CONFIG):
    df = raw.copy()
    df = df[df[C.TARGET_EFF].between(cfg["eff_min"], cfg["eff_max"])]
    df = df[df[C.TARGET_DHU].between(cfg["dhu_min"], cfg["dhu_max"])]
    # deterministic chronological order, tie-break by line id
    df = df.sort_values([C.DATE_COL, C.LINE_ID], kind="mergesort").reset_index(drop=True)
    return df


# --------------------------------------------------------------- feature build
def engineer(df, cfg=C.CONFIG):
    """Add the 7 engineered pre-shift features. History is per-line, computed
    over RETAINED (already-cleaned) records, shifted by one record so only
    strictly past values enter -- a deployed system must reproduce the same
    cleaning exclusions to see the same inputs."""
    df = df.copy()
    df["DayOfWeek"] = df[C.DATE_COL].dt.dayofweek
    df["Month"] = df[C.DATE_COL].dt.month
    mp = df["ManPowerPresent"].replace(0, np.nan)
    df["SMVPerWorker"] = df["SMV"] / mp

    w = cfg["roll_window"]
    g = df.groupby(C.LINE_ID, sort=False)
    df["EffLag1"] = g[C.TARGET_EFF].shift(1)
    df["DHULag1"] = g[C.TARGET_DHU].shift(1)
    # rolling mean of the last `w` retained records, shifted by 1 (strictly past)
    df["EffRoll7"] = (g[C.TARGET_EFF]
                      .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean()))
    df["DHURoll7"] = (g[C.TARGET_DHU]
                      .transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean()))
    return df


# --------------------------------------------------------------- split + scale
@dataclass
class Part:
    Xnum: np.ndarray
    Xcat: np.ndarray
    Xord: np.ndarray
    ycols: np.ndarray
    yorig: np.ndarray
    part: pd.DataFrame


@dataclass
class Splits:
    train: Part
    val: Part
    test: Part
    cat_cardinalities: list
    target_scaler: StandardScaler
    num_cols: list
    cat_cols: list


def _vocab(train_series):
    """Map categories to indices 1..K; reserve 0 for unseen/missing."""
    cats = sorted(pd.Series(train_series).dropna().unique().tolist())
    return {c: i + 1 for i, c in enumerate(cats)}


def make_splits(df, cfg=C.CONFIG):
    df = engineer(df, cfg)
    n = len(df)
    n_tr = int(n * cfg["train_frac"])
    n_vl = int(n * cfg["val_frac"])
    tr = df.iloc[:n_tr].copy()
    vl = df.iloc[n_tr:n_tr + n_vl].copy()
    te = df.iloc[n_tr + n_vl:].copy()

    num = C.NUM_COLS
    # train medians for imputation
    med = tr[num].median(numeric_only=True)

    # numeric scaler (RobustScaler) fit on train
    xsc = RobustScaler()
    xsc.fit(tr[num].fillna(med))

    def num_mat(part):
        return xsc.transform(part[num].fillna(med))

    # target scaler fit on train
    tsc = StandardScaler()
    ytr_orig = tr[[C.TARGET_EFF, C.TARGET_DHU]].values
    tsc.fit(ytr_orig)

    def y_mats(part):
        yo = part[[C.TARGET_EFF, C.TARGET_DHU]].values
        return tsc.transform(yo), yo

    # categorical vocabularies (train only)
    vocabs = {c: _vocab(tr[c]) for c in C.CAT_COLS}
    cards = [len(vocabs[c]) + 1 for c in C.CAT_COLS]      # +1 reserved index

    def cat_idx(part):
        cols = []
        for c in C.CAT_COLS:
            cols.append(part[c].map(vocabs[c]).fillna(0).astype(int).values)
        return np.column_stack(cols)

    def ordinal(part):
        # numeric (scaled) + ordinal-coded cats appended -> ridge/xgboost view
        return np.column_stack([num_mat(part), cat_idx(part).astype(float)])

    def build(part):
        yc, yo = y_mats(part)
        return Part(Xnum=num_mat(part), Xcat=cat_idx(part), Xord=ordinal(part),
                    ycols=yc, yorig=yo, part=part)

    return Splits(train=build(tr), val=build(vl), test=build(te),
                  cat_cardinalities=cards, target_scaler=tsc,
                  num_cols=num, cat_cols=C.CAT_COLS)


# --------------------------------------------------------------- native cats (GBT)
def native_cat_frames(splits):
    """Frames for LightGBM/CatBoost native categorical handling: scaled numeric
    columns + raw category columns as pandas 'category' dtype with a shared
    train vocabulary."""
    tr, vl, te = splits.train.part, splits.val.part, splits.test.part
    num = splits.num_cols
    med = tr[num].median(numeric_only=True)

    def frame(part):
        X = part[num].fillna(med).copy()
        for c in splits.cat_cols:
            X[c] = pd.Categorical(part[c],
                                  categories=sorted(pd.Series(tr[c]).dropna().unique()))
        return X

    return frame(tr), frame(vl), frame(te), splits.cat_cols


if __name__ == "__main__":
    df = clean(load_raw())
    sp = make_splits(df)
    print("rows:", len(df),
          "| train/val/test:", len(sp.train.part), len(sp.val.part), len(sp.test.part))
    print("num features:", len(sp.num_cols), "| cat cardinalities:", sp.cat_cardinalities)
    print("total model inputs:", len(sp.num_cols) + len(sp.cat_cols))
    print("test lines:", sp.test.part[C.LINE_ID].nunique(),
          "| test dates:", sp.test.part[C.DATE_COL].nunique())
