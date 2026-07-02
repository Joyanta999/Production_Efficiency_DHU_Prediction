#!/usr/bin/env python3
"""
Orchestrator. Runs the full study and writes results to ../results/.

Steps:
  1. load + clean + split
  2. (optional) per-architecture HPO on the validation split  [--tune]
  3. train every deep model over N_SEEDS seeds (MTL fixed 0.5/0.5)
  4. single-task ablation per architecture
  5. MTL uncertainty-weighting ablation on the lead network
  6. references: ridge, xgboost, lightgbm, catboost + naive baselines
  7. line-clustered + iid bootstrap CIs and pairwise diffs
  8. Holm-corrected paired seed tests
  9. validation-tuned breach-alert + min-history bins
 10. rolling-origin evaluation for ALL main models
Outputs:
  results/final_results.json     (everything, machine-readable)
  results/table1_models.csv      (deep + tree models, mean +/- SD over seeds)
  results/table2_baselines.csv   (non-deep references)
  results/rolling_origin.csv     (avg R2 per model per fold)

Run:
  VISTAQ_CSV=../data/VistaQDailyProduction.csv python run_all.py
  # quick smoke test:
  N_SEEDS=1 python run_all.py --models MLP --no-tune --quick
"""
import argparse
import json
import os
import numpy as np
import pandas as pd

import config as C
from data import load_raw, clean, make_splits
from references import run_all_references

# Deep-learning imports are optional: if torch is unavailable the run still
# produces the references / baselines / bootstrap / rolling-origin (trees).
try:
    from models import MODEL_ZOO
    from train import train_model
    HAS_TORCH = True
except Exception as _e:                       # torch not installed
    MODEL_ZOO = {}
    train_model = None
    HAS_TORCH = False
    print("WARNING: torch unavailable (%s); deep models will be skipped." % _e)
from evaluate import (summarize, bootstrap, holm, paired_seed_tests,
                      tuned_breach_alert, nominal_breach_alert, min_history_bins)
from rolling_origin import rolling_origin


def mean_sd(rows, key):
    a = np.array([r[key] for r in rows], float)
    return float(a.mean()), float(a.std(ddof=1) if len(a) > 1 else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODEL_ZOO))
    ap.add_argument("--tune", dest="tune", action="store_true", default=False)
    ap.add_argument("--no-tune", dest="tune", action="store_false")
    ap.add_argument("--quick", action="store_true", help="tiny budget smoke test")
    ap.add_argument("--skip-rolling", action="store_true")
    args = ap.parse_args()

    seeds = C.SEEDS[:C.N_SEEDS]
    dl = dict(C.DL)
    if args.quick:
        dl["max_epochs"] = 2; dl["patience"] = 2

    print("Loading data:", C.CONFIG["data_path"])
    df = clean(load_raw())
    sp = make_splits(df)
    yte = sp.test.yorig
    lines_te = sp.test.part[C.LINE_ID].values
    OUT = dict(config={k: C.CONFIG[k] for k in
                       ("eff_min", "eff_max", "dhu_min", "dhu_max",
                        "train_frac", "val_frac", "roll_window")},
               seeds=seeds, n_test=len(yte),
               n_test_lines=int(sp.test.part[C.LINE_ID].nunique()),
               n_test_dates=int(sp.test.part[C.DATE_COL].nunique()))

    # ---- 2. optional per-arch tuning
    hp_by_model = {}
    if args.tune:
        from tune import tune_all
        lead = [m for m in ("TabTransformer", "FTTransformer", "ResNet") if m in args.models]
        tres = tune_all(sp, models=lead, n_trials=2 if args.quick else 8)
        OUT["hpo"] = tres
        hp_by_model = {m: tres[m]["best_hp"] for m in tres}

    # ---- 3 & 4. deep models: MTL + single-task ablation
    deep_rows = {}; per_seed_avg = {}; ablation = {}
    test_preds_for_boot = {}
    val_dhu_preds = {}; test_dhu_preds = {}
    lead_model = None
    for name in args.models:
        runs = [train_model(name, sp, s, hp=hp_by_model.get(name), dl=dl) for s in seeds]
        metrics = [summarize(yte, r["pred_eff"], r["pred_dhu"]) for r in runs]
        deep_rows[name] = {k: mean_sd(metrics, k) for k in metrics[0]}
        deep_rows[name]["epochs"] = float(np.mean([r["epochs"] for r in runs]))
        deep_rows[name]["n_params"] = runs[0]["n_params"]
        per_seed_avg[name] = [m["avg_r2"] for m in metrics]
        # seed-42 (or first) predictions for the bootstrap + breach alert
        test_preds_for_boot[name] = (runs[0]["pred_eff"], runs[0]["pred_dhu"])

        # single-task ablation (both heads, separately)
        st_e = [summarize(yte, train_model(name, sp, s, hp=hp_by_model.get(name),
                                           task_weights=(1.0, 0.0), dl=dl)["pred_eff"],
                          yte[:, 1])["eff_r2"] for s in seeds]
        st_d = [summarize(yte, yte[:, 0],
                          train_model(name, sp, s, hp=hp_by_model.get(name),
                                      task_weights=(0.0, 1.0), dl=dl)["pred_dhu"])["dhu_r2"]
                for s in seeds]
        ablation[name] = dict(
            eff_mtl=mean_sd(metrics, "eff_r2"), eff_single=(float(np.mean(st_e)),
                                                            float(np.std(st_e, ddof=1) if len(st_e) > 1 else 0)),
            dhu_mtl=mean_sd(metrics, "dhu_r2"), dhu_single=(float(np.mean(st_d)),
                                                            float(np.std(st_d, ddof=1) if len(st_d) > 1 else 0)))
    OUT["deep_models"] = deep_rows
    OUT["single_task_ablation"] = ablation

    # ---- 5. MTL uncertainty weighting ablation on the lead network
    lead = next((m for m in ("FTTransformer", "TabTransformer") if m in args.models), None)
    if lead:
        unc = [summarize(yte, *(lambda r: (r["pred_eff"], r["pred_dhu"]))(
                train_model(lead, sp, s, hp=hp_by_model.get(lead), mtl="uncertainty", dl=dl)))
               for s in seeds]
        OUT["mtl_uncertainty"] = dict(model=lead,
                                      fixed_avg_r2=deep_rows[lead]["avg_r2"],
                                      uncertainty_avg_r2=mean_sd(unc, "avg_r2"))

    # ---- 6. references
    refs = run_all_references(sp, seed=seeds[0])
    OUT["references"] = {n: summarize(yte, r["pred_eff"], r["pred_dhu"])
                         for n, r in refs.items()}
    for n, r in refs.items():
        test_preds_for_boot[n] = (r["pred_eff"], r["pred_dhu"])
        if r.get("val_dhu") is not None:
            val_dhu_preds[n] = r["val_dhu"]; test_dhu_preds[n] = r["pred_dhu"]

    # ---- 7. bootstrap
    pairs = [("LightGBM", "XGBoost"), ("CatBoost", "XGBoost"),
             ("XGBoost", lead or "TabTransformer"), ("Ridge", lead or "TabTransformer"),
             ("XGBoost", "Persistence"), (lead or "TabTransformer", "RollingMean7"),
             ("RollingMean7", "Persistence"), ("XGBoost", "RollingMean7")]
    boot_models = {k: v for k, v in test_preds_for_boot.items()}
    OUT["bootstrap_cluster"] = bootstrap(yte, boot_models, lines_te, pairs,
                                         B=200 if args.quick else C.BOOTSTRAP_B)
    OUT["bootstrap_iid"] = bootstrap(yte, boot_models, np.arange(len(yte)), pairs,
                                     B=200 if args.quick else C.BOOTSTRAP_B)

    # ---- 8. Holm over paired seed tests (avg R2)
    raw = paired_seed_tests(per_seed_avg, "avg_r2")
    OUT["seed_tests"] = dict(raw=raw, holm=holm(raw)) if raw else {}

    # ---- 9. breach alert + min history
    yvl_dhu = sp.val.yorig[:, 1]
    # add deep lead to breach alert if its val preds available -> retrain once on val
    OUT["breach_tuned"] = tuned_breach_alert(yte[:, 1], yvl_dhu, test_dhu_preds,
                                             val_dhu_preds, C.BREACH_THRESHOLDS)
    OUT["breach_nominal"] = nominal_breach_alert(yte[:, 1], test_dhu_preds,
                                                 C.BREACH_THRESHOLDS)
    mh_models = {k: test_preds_for_boot[k]
                 for k in ("XGBoost", "Persistence", "RollingMean7")
                 if k in test_preds_for_boot}
    OUT["min_history"] = min_history_bins(df, sp.test.part, yte, mh_models, C.LINE_ID)

    # ---- 10. rolling origin for ALL main models
    if not args.skip_rolling:
        deep_for_roll = [m for m in ("TabTransformer", "FTTransformer") if m in args.models]
        OUT["rolling_origin"] = rolling_origin(
            df, deep_models=tuple(deep_for_roll),
            tree_models=("xgb", "lgbm", "cat", "ridge"),
            dl_seed=seeds[0], hp_by_model=hp_by_model)

    # ---- write
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    with open(os.path.join(C.RESULTS_DIR, "final_results.json"), "w") as f:
        json.dump(OUT, f, indent=1, default=str)

    # tidy tables
    t1 = []
    for n, d in deep_rows.items():
        t1.append(dict(model=n, eff_r2=d["eff_r2"][0], dhu_r2=d["dhu_r2"][0],
                       avg_r2=d["avg_r2"][0], avg_r2_sd=d["avg_r2"][1],
                       n_params=d["n_params"]))
    for n in ("LightGBM", "XGBoost", "CatBoost"):
        if n in OUT["references"]:
            s = OUT["references"][n]
            t1.append(dict(model=n + " (ref.)", eff_r2=s["eff_r2"], dhu_r2=s["dhu_r2"],
                           avg_r2=s["avg_r2"], avg_r2_sd=0.0, n_params=""))
    pd.DataFrame(t1).sort_values("avg_r2", ascending=False).to_csv(
        os.path.join(C.RESULTS_DIR, "table1_models.csv"), index=False)

    t2 = [dict(model=n, **{k: OUT["references"][n][k] for k in
                           ("eff_mae", "eff_rmse", "eff_r2", "dhu_mae", "dhu_rmse",
                            "dhu_r2", "avg_r2")})
          for n in ("TrainingMean", "Persistence", "RollingMean7", "Ridge")
          if n in OUT["references"]]
    pd.DataFrame(t2).to_csv(os.path.join(C.RESULTS_DIR, "table2_baselines.csv"), index=False)

    if "rolling_origin" in OUT:
        rr = []
        for fold in OUT["rolling_origin"]:
            row = dict(fold=fold["fold"], n_test=fold["n_test"],
                       test_start=fold["test_start"], test_end=fold["test_end"])
            row.update(fold["models"])
            rr.append(row)
        pd.DataFrame(rr).to_csv(os.path.join(C.RESULTS_DIR, "rolling_origin.csv"), index=False)

    print("\nDone. Wrote results to", C.RESULTS_DIR)
    print(pd.DataFrame(t1).sort_values("avg_r2", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
