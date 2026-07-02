# FinalPaperV1 — revision log (response to the "critical & important" review items)

This revision rebuilds the study pipeline from scratch and adds the baselines /
ablations the review asked for. It changes **method and code**, not the headline
story. Result tables in the manuscript still show the v5 numbers and are flagged
with a visible *Revision note*; they are refreshed automatically from
`results/final_results.json` once the real dataset is supplied (see README).

## What was changed and why

### Critical items (threaten the central "trees beat networks / MTL adds nothing" claims)

1. **Per-architecture hyperparameter tuning.** v5 shared one configuration
   across all networks, so "deep learning loses" was open to "untuned networks
   vs tuned trees." `tune.py` does a validation-only random search per
   architecture (lead networks by default); `run_all.py --tune` wires the best
   configs into the final seed ensemble and the rolling-origin folds.

2. **MTL objective actually tuned.** v5 fixed the loss at 0.5/0.5 and then
   declared MTL unhelpful. We added a homoscedastic uncertainty-weighted loss
   (Kendall et al. [25], `models.MTLLoss(mode="uncertainty")`) and report it as
   an ablation on the lead network, so the MTL hypothesis is tested in its
   strongest form before being rejected.

3. **Rolling-origin for every main model.** v5 ran the five expanding-window
   folds for XGBoost only. `rolling_origin.py` now retrains and scores *all*
   main models per fold — deep, XGBoost, LightGBM, CatBoost, ridge, persistence,
   rolling mean — so window-robustness of the ranking is checked for everyone.

### Important items

4. **More seeds.** Default raised from 5 to 15 (`config.SEEDS`, `N_SEEDS`),
   giving the Holm family real power instead of near-guaranteed non-significance.

5. **Leakage literature.** Related Work now cites Kaufman et al. [27] and
   Kapoor & Narayanan [26], anchoring the paper's central contribution (the
   feature-admissibility audit) in the established leakage literature.

6. **Reproducibility without the proprietary data.** `make_synthetic.py`
   generates a schema-faithful synthetic export so the entire pipeline runs
   end-to-end for review even though the real data cannot be shared.

### Explicitly requested additions

7. **CatBoost** reference with native categorical handling (`references.run_catboost`).

8. **RTDL-style strong tabular DL baselines:** an **FT-Transformer** (numeric
   features linearly tokenised, categoricals embedded, `[CLS]` read-out) and a
   tabular **ResNet**, both in `models.py`, both multi-task. Citing Gorishniy
   et al. [24].

9. **Learned categorical embeddings replace ordinal-only encoding** on every
   neural model (`models.TabularInput`): each of factory / building / buyer gets
   its own embedding table instead of an ordinal code fed as a magnitude. This
   removes the encoding handicap the v5 paper listed as a limitation, on the
   network side directly.

## Exact manuscript edits (applied by `paper/patch_paper_finalv1.py`)

The patch script transforms `build_paper_v5.js` → `build_paper_finalv1.js` with
nine asserted string replacements (each fails loudly if its anchor moves):

| # | Section | Change |
|---|---------|--------|
| 1 | build infra | figure path → `FIG_DIR` env / `./figures` (was a dead absolute path) |
| 2 | References | added [23] CatBoost, [24] RTDL/FT-Transformer, [25] Kendall, [26] Kapoor & Narayanan, [27] Kaufman et al. |
| 3 | II-C Related Work | RTDL strong baselines, CatBoost, and the leakage literature |
| 4 | IV-A Framework | states categoricals enter as learned embeddings, not ordinal codes |
| 5 | IV-C References | adds CatBoost + FT-Transformer + ResNet to the model ladder |
| 6 | IV-C Encoding | rewrites the ordinal-handicap rationale around learned embeddings |
| 7 | IV-C Ablation | adds uncertainty-weighted MTL loss + per-architecture tuning |
| 8 | V Setup | fifteen seeds; rolling-origin for all model classes |
| 9 | VI Results | visible *Revision note* that tables are pending the re-run |

`OUT_DOCX` path also de-hardcoded.

## How the numbers get refreshed

The result tables (I–V), inline CIs, and the abstract figures are **not**
hand-edited to invented numbers. After running the pipeline on the real CSV,
`results/final_results.json` holds every value the manuscript needs
(`deep_models`, `references`, `bootstrap_cluster/iid`, `single_task_ablation`,
`mtl_uncertainty`, `breach_tuned`, `min_history`, `rolling_origin`, `hpo`).
Wire those into the table arrays at the top of `build_paper_finalv1.js`
(`T1R`, `T2R`, `T4R`, `T5R`, …) and re-run `node build_paper_finalv1.js`.

## What could not be done in this environment

The deep models were **not executed here**: PyTorch's CPU wheel is blocked by
the sandbox proxy and the default wheel exceeds the time limit. The non-deep
path (data, CatBoost/XGBoost/LightGBM/ridge, cluster+iid bootstrap, breach
alert, min-history, rolling-origin for trees) was run end-to-end on synthetic
data and verified. The deep code compiles cleanly and is ready to run in any
environment with `torch` installed.
