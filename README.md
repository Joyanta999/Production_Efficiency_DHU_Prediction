# FinalPaperV1 — rebuilt pipeline + revised manuscript

A from-scratch reimplementation of the v5 leakage-audited study, extended with
the reviewer-requested baselines and ablations. See
`docs/REVISIONS_FinalPaperV1.md` for the full change log.

## Layout

```
FinalPaperV1/
├── code/
│   ├── config.py            # paths, filters, seeds, model params
│   ├── data.py              # load/clean/feature-engineer/split, train-only scaling
│   ├── models.py            # MTL nets w/ learned embeddings: MLP, DeepMLP, CNN1D,
│   │                        #   TabTransformer, BiLSTM + RTDL FT-Transformer & ResNet;
│   │                        #   fixed vs Kendall uncertainty-weighted loss
│   ├── train.py             # training loop (AdamW, early stop, single-task ablation)
│   ├── references.py        # Ridge, XGBoost, LightGBM, CatBoost + naive baselines
│   ├── evaluate.py          # cluster+iid bootstrap, Holm, breach alert, history bins
│   ├── rolling_origin.py    # rolling-origin for EVERY main model
│   ├── tune.py              # per-architecture validation HPO
│   ├── run_all.py           # orchestrator -> results/
│   ├── make_synthetic.py    # schema-faithful synthetic data for testing
│   └── requirements.txt
├── data/                    # put VistaQDailyProduction.csv here (or set VISTAQ_CSV)
├── results/                 # JSON + CSV outputs (final_results.json, table*.csv, ...)
├── paper/
│   ├── build_paper_v5.js        # original v5 manuscript builder (unchanged)
│   ├── build_paper_finalv1.js   # REVISED builder (new refs/methods/related work)
│   ├── patch_paper_finalv1.py   # produces build_paper_finalv1.js from v5 (asserted edits)
│   ├── IEEE_paper_FinalPaperV1.docx / .pdf   # built with PLACEHOLDER figures
│   └── figures/                 # placeholder figures (replace with real ones)
└── docs/
    ├── REVISIONS_FinalPaperV1.md
    └── FIXES_V5.md
```

## Run it

```bash
cd FinalPaperV1/code
pip install -r requirements.txt          # use --break-system-packages if needed

# 1) with your real data:
VISTAQ_CSV=/path/to/VistaQDailyProduction.csv python run_all.py --tune

# 2) or smoke-test on synthetic data first:
python make_synthetic.py --out ../data/VistaQDailyProduction.csv
VISTAQ_CSV=../data/VistaQDailyProduction.csv python run_all.py --no-tune

# quick low-budget pass:
N_SEEDS=2 VISTAQ_CSV=../data/VistaQDailyProduction.csv python run_all.py --quick --skip-rolling
```

Outputs land in `results/`: `final_results.json` (everything),
`table1_models.csv`, `table2_baselines.csv`, `rolling_origin.csv`.

> If `torch` is not installed the run still produces all non-deep results and
> prints a warning; install `torch` to include the neural models.

## Rebuild the manuscript

```bash
cd FinalPaperV1/paper
npm install docx
# replace placeholder PNGs in ./figures with the real figures, then:
FIG_DIR=./figures node build_paper_finalv1.js      # -> IEEE_paper_FinalPaperV1.docx
```

To refresh the metric tables with real numbers, copy the relevant values from
`results/final_results.json` into the table arrays (`T1R`, `T2R`, `T4R`, `T5R`)
near the top of `build_paper_finalv1.js` and rebuild.

## Status of this build

The shipped `.docx`/`.pdf` use **placeholder figures** and still carry the **v5
metric numbers** (flagged in-text). The neural models were not executed in the
build sandbox (no `torch` available there); the tree / linear / bootstrap /
rolling-origin path was verified end-to-end on synthetic data.
