# Table IV (v5) — consistent 3-dp rounding, same seeds/ddof as Table I

| Architecture | Task | Single-task R2 | Multi-task R2 |
|---|---|---|---|
| MLP | eff | 0.256 ± 0.090 | 0.309 ± 0.101 |
| MLP | dhu | 0.680 ± 0.096 | 0.673 ± 0.070 |
| DeepMLP | eff | 0.018 ± 0.127 | -0.049 ± 0.150 |
| DeepMLP | dhu | 0.607 ± 0.031 | 0.536 ± 0.060 |
| TabTransformer | eff | 0.360 ± 0.051 | 0.327 ± 0.059 |
| TabTransformer | dhu | 0.767 ± 0.008 | 0.763 ± 0.014 |
| CNN1D | eff | 0.345 ± 0.027 | 0.332 ± 0.031 |
| CNN1D | dhu | 0.555 ± 0.038 | 0.548 ± 0.050 |
| BiLSTM | eff | 0.232 ± 0.093 | 0.110 ± 0.223 |
| BiLSTM | dhu | 0.620 ± 0.078 | 0.607 ± 0.071 |
