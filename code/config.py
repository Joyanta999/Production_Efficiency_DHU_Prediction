#!/usr/bin/env python3
"""
Central configuration for the FinalPaperV1 rebuild.

This is a from-scratch reimplementation of the v5 leakage-audited protocol,
extended (per review) with:
  * CatBoost reference (native categorical handling)
  * RTDL-style strong tabular DL baselines: FT-Transformer and ResNet
  * learned categorical EMBEDDINGS on every neural model (replacing the
    ordinal-only encoding used in v5)
  * rolling-origin evaluation for EVERY main model (not only XGBoost)
  * per-architecture hyperparameter search (validation-tuned)
  * Kendall-style uncertainty-weighted MTL loss ablation
  * more seeds (default 15)

Everything is driven by the dataset path in CONFIG['data_path'] (override with
the VISTAQ_CSV env var). When no real CSV is available you can generate a
schema-faithful synthetic file with make_synthetic.py and point CONFIG at it.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                      # FinalPaperV1/
RESULTS_DIR = os.path.join(ROOT, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --------------------------------------------------------------- data + protocol
CONFIG = dict(
    # Path to the raw daily-production export. Override: VISTAQ_CSV=... python run_all.py
    data_path=os.environ.get(
        "VISTAQ_CSV",
        os.path.join(ROOT, "data", "VistaQDailyProduction.csv"),
    ),
    # cleaning filters (percentage points)
    eff_min=5.0, eff_max=130.0,
    dhu_min=0.1, dhu_max=50.0,
    # chronological split fractions
    train_frac=0.70, val_frac=0.15,   # test_frac = 1 - train - val
    # per-line history
    roll_window=7,
    # only factories whose name carries the corporate "Ltd" marker (matches v5)
    factory_filter="Ltd",
)

# Seeds. v5 used 5; the review asked for more power -> default 15.
SEEDS = [42, 123, 2024, 7, 99, 11, 256, 1337, 2718, 31415,
         170, 8, 64, 512, 90210]

# How many seeds to actually use (env override for quick runs / smoke tests).
N_SEEDS = int(os.environ.get("N_SEEDS", len(SEEDS)))

# --------------------------------------------------------------- columns
# Raw target columns (as they appear in the export).
TARGET_EFF = "AchievedEfficiency"
TARGET_DHU = "dhu"
LINE_ID = "WorkspaceLineId"
DATE_COL = "Date"

# Categorical fields handled natively by trees and via learned embeddings by nets.
CAT_COLS = ["WorkspaceFactoryName", "WorkspaceBuildingName", "BuyerName"]

# 12 plan-time numerical fields knowable at the morning muster.
PLAN_NUM_COLS = [
    "SMV", "SampleSMV", "CM", "DayTarget", "IETarget", "TargetEfficiency",
    "ManPowerPresent", "PlannedIronMan", "PlannedHelper", "PlannedOperator",
    "PlannedHours", "RunningWorkDay",
]

# 7 engineered pre-shift features (built in data.py): DayOfWeek, Month,
# SMVPerWorker, EffLag1, DHULag1, EffRoll7, DHURoll7.
ENG_NUM_COLS = [
    "DayOfWeek", "Month", "SMVPerWorker",
    "EffLag1", "DHULag1", "EffRoll7", "DHURoll7",
]

NUM_COLS = PLAN_NUM_COLS + ENG_NUM_COLS          # 19 numeric
# total model inputs = 19 numeric + 3 categorical = 22

# --------------------------------------------------------------- GBT params (v5)
XGB_PARAMS = dict(
    n_estimators=2000, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    n_jobs=0, tree_method="hist",
)
LGB_PARAMS = dict(
    n_estimators=2000, learning_rate=0.05, num_leaves=63,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
    reg_lambda=1.0, verbose=-1,
)
CAT_PARAMS = dict(
    iterations=2000, learning_rate=0.05, depth=6,
    l2_leaf_reg=3.0, loss_function="RMSE", verbose=False,
    allow_writing_files=False,
)
RIDGE_ALPHAS = [0.1, 1, 10, 100, 1000]           # chosen on validation
EARLY_STOP_ROUNDS = 50

# --------------------------------------------------------------- DL training
DL = dict(
    batch_size=64, max_epochs=150, patience=20,
    lr=1e-3, weight_decay=1e-4, grad_clip=1.0,
    lr_factor=0.5, lr_patience=8, lr_min=1e-6,
    seq_len=7,                                    # BiLSTM per-line window
)

BREACH_THRESHOLDS = [10.0, 15.0]
BOOTSTRAP_B = 2000
ROLLING_FOLDS = 5
