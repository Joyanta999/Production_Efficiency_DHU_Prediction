# FV2 / v5 revision — fourth-round audit and fixes (June 2026)

Reference: `../Fable5` (v4 study), paper `paper/IEEE_paper_Fable5_v4.docx`.
The v4 audit found four substantive vulnerabilities and several smaller ones.
Everything below is fixed in `IEEE_paper_Fable5_v5.docx`, with new evidence in
`FV2/results/` produced by `FV2/code/v5_experiments.py`.

## Substantive

**1. The v4 bootstrap assumed i.i.d. test rows (statistical vulnerability).**
The 1,111 test rows cluster within 141 production lines; line-days of one line
share level and residual correlation, so row-level resampling understates
uncertainty, and the tightest headline claim (LightGBM > XGBoost,
CI [0.005, 0.028]) was at risk.
*Fix:* primary instrument is now a cluster bootstrap that resamples whole
lines (B = 2,000, same RNG protocol); the i.i.d. bootstrap is reported
alongside. **Outcome: every headline claim survives.** LightGBM−XGBoost
clustered CI [0.004, 0.030] (row-level [0.005, 0.028]); XGBoost−TabTransformer
[0.006, 0.061]; ridge−TabTransformer still includes 0 ([−0.008, 0.036]).
Clustered intervals are wider, as expected, but no conclusion flips.
Clustering by date is impossible (nine test dates) and the paper says so.
Sec. V and VI-A rewritten; Limitations updated.

**2. Missing rolling-mean baseline (the strongest naive forecast).**
Predicting each target with its own 7-day per-line rolling mean — literally
the EffRoll7/DHURoll7 input columns passed through — scores avg R² 0.520
(eff 0.307, DHU 0.733), beating persistence (0.501) and landing within 0.025
of the best network's five-seed mean (0.545 ± 0.035).
*Fix:* added everywhere a baseline appears: Table II row; Table V row
(nominal alert: F1 .822@10, .745@15; tuned: .818/.738 — weakest alert, and
the paper explains why smoothing hurts detection); Sec. V baseline
definitions; Sec. VI-B rewritten around it (TabT−rollmean clustered CI
[0.032, 0.123] excludes 0, so the network adds something, but the paper now
calls the margin modest explicitly; XGB−rollmean [0.061, 0.165]);
min-history bins; abstract, contributions, practitioner guidance, conclusion.

**3. Confidentiality leak in shippable artifacts.**
`results/robustness.json` carried real factory names in its `per_factory`
keys, and `docs/FIXES.md` named a buyer with its share, while the README
declared only `anonymization_key.json` confidential.
*Fix:* `FV2/results/robustness_v5_anon.json` is the shippable version
(Factory A/B/C keys; script asserts no real factory or buyer string survives
anywhere in the JSON). This FIXES file names no factory or buyer. The v5
Data Availability statement commits to the re-audit and excludes the key
from any release. **The original `../results/robustness.json` and
`../docs/FIXES.md` must NOT ship as supplementary material.**

**4. Factual errors in the cleaning description.**
"The DHU floor removes 165 genuinely zero-defect line-days" — 165 is the raw
count; after the efficiency filter only 91 zero-DHU rows remain for the DHU
floor to remove. "About 1.5 workers" — the median within-line SD of
ManPowerPresent is 1.61.
*Fix:* Sec. III-B now says 91 (165 in the raw export); Sec. III-C says 1.6.
Both recomputed and stored in `v5_results.json["facts"]`.

## Smaller

**5. "Yesterday" vs previous retained record.** Lags are previous-record
within line, computed after the cleaning filters, so a lag-1 can reach back
across weekends, idle days, and filtered records — and a deployed system must
reproduce the exclusions. Sec. III-C and the persistence definition in Sec. V
now say "most recent retained record" and state the deployment requirement.

**6. Underpowered seed tests.** Five seeds under a seventeen-comparison Holm
family near-guarantees non-significance. Sec. V now states the low power and
that non-significance is absence of evidence, not evidence of absence.

**7. Min-history nuance.** Persistence scores 0.58 (and the rolling mean
0.47) in the 5-9-record bin where XGBoost collapses to −4.1; "a month of
history is the price of admission" now applies explicitly to the learned
models, with persistence recommended for newly commissioned lines. Sec. VI-H.

**8. Test-window precision.** The 4-13 January window contains nine
production dates (one date has no records); "ten days" corrected in VI-H and
Limitations, and the nine-date count is the stated reason date-clustered
resampling is impossible.

**9. Rounding consistency.** MLP efficiency R² mean is 0.30949; Table I and
Table IV both now print 0.309 (Table I previously said 0.310).
`FV2/results/ablation_table4_v5.md` regenerates the ablation table with the
same rounding and ddof as Table I.

## Verification

`v5_experiments.py` reproduces the v4 numbers exactly before extending them
(same seed-42 predictions, same i.i.d. CIs to 3 dp, same tuned breach F1s),
so the new claims are drop-in replacements, not a re-run that happened to
agree. XGBoost 3.2.0 (CPU wheel) and LightGBM 4.6.0 were verified to
reproduce the archived per-seed CSVs.
