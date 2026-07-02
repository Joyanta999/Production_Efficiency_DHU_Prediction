#!/usr/bin/env python3
"""Builds build_paper_v5.js from the v4 builder by applying every v5 edit in
one pass (single atomic write). Each replacement must match exactly once."""
import io, os, sys

SRC = '/sessions/serene-intelligent-archimedes/mnt/Fable5/paper/build_paper_v4.js'
DST = '/sessions/serene-intelligent-archimedes/mnt/Fable5/FV2/paper/build_paper_v5.js'
s = io.open(SRC, encoding='utf-8').read()

R = []
def rep(old, new):
    R.append((old, new))

# --- paths -------------------------------------------------------------
rep("/sessions/stoic-youthful-euler/mnt/Fable5/figures",
    "/sessions/serene-intelligent-archimedes/mnt/Fable5/figures")
rep("/sessions/stoic-youthful-euler/mnt/outputs/IEEE_paper_Fable5_v4.docx",
    "/sessions/serene-intelligent-archimedes/mnt/Fable5/FV2/paper/IEEE_paper_Fable5_v5.docx")

# --- tables ------------------------------------------------------------
rep("'18.93 ± 1.40', '0.310 ± 0.101',", "'18.93 ± 1.40', '0.309 ± 0.101',")
rep("  ['Per-line persistence', '11.90', '19.35', '0.281', '1.63', '3.36', '0.721', '0.501'],",
    "  ['Per-line persistence', '11.90', '19.35', '0.281', '1.63', '3.36', '0.721', '0.501'],\n"
    "  ['Per-line 7-day rolling mean', '13.65', '19.01', '0.307', '1.74', '3.29', '0.733', '0.520'],")
rep("  ['MLP', '0.310 ± 0.101', '0.256 ± 0.090',",
    "  ['MLP', '0.309 ± 0.101', '0.256 ± 0.090',")
rep("  ['Persistence', '0.810', '0.848', '0.829', '0.800', '0.780', '0.790'],",
    "  ['Persistence', '0.810', '0.848', '0.829', '0.800', '0.780', '0.790'],\n"
    "  ['Rolling mean (7-day)', '0.779', '0.870', '0.822', '0.845', '0.667', '0.745'],")

# --- abstract ----------------------------------------------------------
rep("all results face naive-mean, persistence, linear, and gradient-boosted comparisons.",
    "all results face naive-mean, persistence, rolling-mean, linear, and gradient-boosted comparisons.")
rep("margins whose bootstrap confidence intervals over test rows exclude zero — "
    "a plain ridge regression is statistically indistinguishable from the best network, and "
    "single-task ablations across the architectures find no reliable benefit from the shared encoder.",
    "margins whose bootstrap confidence intervals, clustered by production line to respect the "
    "correlation of test rows within lines, exclude zero — a plain ridge regression is statistically "
    "indistinguishable from the best network, the strongest network clears a raw rolling-mean "
    "passthrough of its own input features only narrowly, and single-task ablations across the "
    "architectures find no reliable benefit from the shared encoder.")

# --- introduction ------------------------------------------------------
rep("against five reference points: a naive mean, a per-line persistence forecast, a ridge regression, "
    "and two gradient-boosted implementations — XGBoost on the shared encoded features and LightGBM "
    "with native categorical handling — under the identical protocol, with bootstrap confidence "
    "intervals over test rows attached to the headline comparisons.",
    "against six reference points: a naive mean, a per-line persistence forecast, a per-line 7-day "
    "rolling-mean forecast, a ridge regression, and two gradient-boosted implementations — XGBoost on "
    "the shared encoded features and LightGBM with native categorical handling — under the identical "
    "protocol, with line-clustered bootstrap confidence intervals attached to the headline comparisons.")

# --- III-B cleaning ----------------------------------------------------
rep("the DHU floor removes 165 genuinely zero-defect line-days, truncating the easy end of the "
    "quality distribution,",
    "the DHU floor removes the 91 genuinely zero-defect line-days that survive the efficiency filter "
    "(the raw export contains 165 such days), truncating the easy end of the quality distribution,")

# --- III-C features ----------------------------------------------------
rep("and four per-line history features — yesterday’s efficiency and DHU for the same line, plus "
    "seven-day rolling means of each, shifted by one day so that only strictly past values enter. "
    "History is computed within lines, never across them.",
    "and four per-line history features — the line’s most recent retained efficiency and DHU, plus "
    "rolling means of its last seven retained records, shifted by one record so that only strictly "
    "past values enter. “Retained” is doing work in that sentence: the history features are computed "
    "after the cleaning filters of Section III-B, over records rather than calendar days, so a "
    "“lag-1” value is usually yesterday but reaches further back across weekends, idle days, and "
    "filtered records — and a deployed system must reproduce the same exclusions to see the same "
    "inputs. History is computed within lines, never across them.")
rep("median within-line standard deviation of about 1.5 workers",
    "median within-line standard deviation of about 1.6 workers")

# --- V: uncertainty instruments ----------------------------------------
rep("paired t-tests across the five seeds measure robustness to initialization on the fixed split — "
    "nothing more, since the test set never varies — and every such p-value we cite is Holm-adjusted "
    "across the full family of seed comparisons reported anywhere in this paper, ablations included. "
    "Generalization uncertainty is instead quantified with nonparametric bootstrap confidence "
    "intervals (2,000 resamples of the 1,111 test rows), applied to the headline models and to their "
    "pairwise differences; a margin is only called real when its bootstrap CI excludes zero. Where "
    "earlier drafts of this work blurred the two, reviewers were right to object.",
    "paired t-tests across the five seeds measure robustness to initialization on the fixed split — "
    "nothing more, since the test set never varies. Every such p-value we cite is Holm-adjusted "
    "across the full family of seed comparisons reported anywhere in this paper, ablations included, "
    "and with five seeds and seventeen corrected comparisons their power is low: we report them for "
    "transparency, and treat their non-significance as absence of evidence, not evidence of absence. "
    "Generalization uncertainty is instead quantified with nonparametric bootstrap confidence "
    "intervals (B = 2,000), applied to the headline models and to their pairwise differences. The "
    "1,111 test rows are not independent draws — they cluster within 141 production lines, whose "
    "line-days share level and residual correlation — so our primary instrument resamples whole "
    "lines (a cluster bootstrap), which preserves the within-line dependence; the row-level i.i.d. "
    "bootstrap that earlier drafts used is reported alongside and, as expected, gives slightly "
    "narrower intervals. Clustering by date is not an option the data allows: the test window "
    "contains only nine production dates. A margin is only called real when its line-clustered CI "
    "excludes zero. Where earlier drafts of this work blurred initialization and generalization "
    "uncertainty, reviewers were right to object.")

# --- V: baselines ------------------------------------------------------
rep("Two non-learned baselines frame everything: the naive mean, and per-line persistence, which "
    "predicts that each line will repeat yesterday’s efficiency and DHU. On autocorrelated "
    "operational data, persistence is the minimum honest bar for any forecasting claim.",
    "Three non-learned baselines frame everything: the naive mean; per-line persistence, which "
    "predicts that each line will repeat its most recent retained efficiency and DHU; and a per-line "
    "7-day rolling mean, which predicts the average of the line’s last seven retained records. The "
    "rolling mean deserves emphasis, because it is literally one of the model inputs (EffRoll7, "
    "DHURoll7) passed through unchanged: a learned model that fails to beat it is extracting nothing "
    "from its own conditioning set beyond what one smoothed feature already says. On autocorrelated "
    "operational data, these are the minimum honest bars for any forecasting claim.")

# --- Table I title -----------------------------------------------------
rep("Bootstrap 95% CIs Over Test Rows for the Leading Models Are Given in Sec. VI-A.",
    "Line-Clustered Bootstrap 95% CIs for the Leading Models Are Given in Sec. VI-A.")

# --- VI-A headline -----------------------------------------------------
rep("The boosted trees win outright, and the bootstrap places the margin beyond sampling noise: "
    "XGBoost leads the best network by 0.032 average R² (95% CI [0.005, 0.059] over test rows), with "
    "the gap concentrated on efficiency (difference CI [0.01, 0.10]); the seed-paired test points "
    "the same way (raw p = 0.005), though over the full Holm family no seed-level comparison in this "
    "paper survives — the bootstrap intervals, not the seed tests, carry the claim. LightGBM, given "
    "the categorical fields natively, does better still — 0.641 ± 0.005, ahead of XGBoost by a "
    "margin whose CI also excludes zero ([0.005, 0.028]) — which answers the encoding-handicap "
    "objection in the direction least flattering to the networks: removing the handicap helps the "
    "trees. The ridge regression of Table II lands at 0.611, nominally ahead of every network, "
    "though against the TabTransformer specifically the bootstrap difference includes zero "
    "([−0.006, 0.032]);",
    "The boosted trees win outright, and the line-clustered bootstrap places the margin beyond "
    "sampling noise: XGBoost leads the best network by 0.033 average R² (clustered 95% CI "
    "[0.006, 0.061]; row-level [0.005, 0.059]), with the gap concentrated on efficiency (clustered "
    "difference CI [0.010, 0.103]); the seed-paired test points the same way (raw p = 0.005), though "
    "over the full Holm family no seed-level comparison in this paper survives — the bootstrap "
    "intervals, not the seed tests, carry the claim. LightGBM, given the categorical fields "
    "natively, does better still — 0.641 ± 0.005, ahead of XGBoost by a margin whose clustered CI "
    "also excludes zero ([0.004, 0.030]; row-level [0.005, 0.028]) — which answers the "
    "encoding-handicap objection in the direction least flattering to the networks: removing the "
    "handicap helps the trees. The ridge regression of Table II lands at 0.611, nominally ahead of "
    "every network, though against the TabTransformer specifically the clustered bootstrap "
    "difference includes zero ([−0.008, 0.036]);")

# --- VI-B --------------------------------------------------------------
rep("subHead('B. Persistence Is Harder to Beat Than It Looks'),",
    "subHead('B. Persistence and the Rolling Mean Are Harder to Beat Than They Look'),")
rep("Per-line persistence sets a demanding mark: average R² 0.501, within 0.05 of the best deep "
    "model. The TabTransformer improves on it by +0.042 R² on DHU and +0.046 on efficiency — "
    "consistent, but thin. The models that walk away from it are the simple ones: XGBoost clears "
    "the bar by 0.13 average R² and ridge by 0.11, and they are the only two learned models that "
    "undercut its efficiency MAE (11.53 and 11.58 against 11.90). An MAE/RMSE asymmetry is worth "
    "understanding: on DHU, persistence beats every network on MAE (1.63 against 1.76 for the best "
    "of them; only ridge does better, at 1.61) while losing on RMSE and R². Persistence is excellent "
    "on the many routine days when a line simply repeats itself and terrible around changeovers and "
    "disruptions; the learned models give up a little routine-day accuracy to cut the tail errors. "
    "We dwell on this because studies that omit persistence credit their architectures with what is "
    "mostly yesterday’s value of an autocorrelated series. Any procurement of forecasting software "
    "in this domain should demand the persistence comparison in writing.",
    "Per-line persistence sets a demanding mark: average R² 0.501, within 0.05 of the best deep "
    "model. The 7-day rolling mean sets a higher one still: 0.520 (efficiency 0.307, DHU 0.733), "
    "and the difference between the two naive forecasts is itself within noise (clustered CI "
    "[−0.033, 0.074]). That second number should give pause, because the rolling mean is one of the "
    "models’ own input columns passed through unchanged — and it lands within 0.025 of the best "
    "network’s five-seed mean (0.545 ± 0.035). The comparison the framework permits is at the "
    "prediction level: the best network’s test predictions beat the rolling mean with a clustered "
    "CI that excludes zero (+0.079 average R², CI [0.032, 0.123], on the seed-42 model whose "
    "predictions the bootstrap uses), so the network is adding something — but the margin is modest "
    "for 160,066 parameters, and a reader who takes the five-seed mean as the fairer summary will "
    "call it thin. The models that walk away from both naive bars are the simple ones: XGBoost "
    "clears the rolling mean by 0.11 average R² (clustered CI [0.061, 0.165]) and persistence by "
    "0.13, ridge by similar margins, and they are the only two learned models that undercut "
    "persistence’s efficiency MAE (11.53 and 11.58 against 11.90). An MAE/RMSE asymmetry is worth "
    "understanding: on DHU, persistence beats every network on MAE (1.63 against 1.76 for the best "
    "of them; only ridge does better, at 1.61) while losing on RMSE and R²; the rolling mean "
    "smooths away the routine-day sharpness (MAE 1.74) but cuts tail errors (best non-learned DHU "
    "RMSE, 3.29). Persistence is excellent on the many routine days when a line simply repeats "
    "itself and terrible around changeovers; the rolling mean is the opposite trade; the learned "
    "models buy a little of both. We dwell on this because studies that omit such baselines credit "
    "their architectures with what is mostly the recent history of an autocorrelated series. Any "
    "procurement of forecasting software in this domain should demand the persistence and "
    "rolling-mean comparisons in writing.")

# --- VI-E breach -------------------------------------------------------
rep("and reverses at 15%, where persistence wins (0.797 against 0.775 and 0.769).",
    "and reverses at 15%, where persistence wins (0.797 against 0.775 and 0.769). The rolling mean, "
    "strong as a regression baseline, is the weakest alert at both levels (tuned F1 0.818 and "
    "0.738): smoothing seven days of history erases exactly the spikes an alarm exists to catch, a "
    "reminder that the regression and detection rankings need not agree.")

# --- VI-H --------------------------------------------------------------
rep("Our chronological split tests on 4-13 January 2026 — ten days, and we say so plainly.",
    "Our chronological split tests on 4-13 January 2026 — a ten-day span containing nine production "
    "dates, and we say so plainly.")
rep("then jumps to 0.61 at 20-59 records and 0.70 beyond 60. Roughly a month of per-line history is "
    "the price of admission, Factory C’s lines (767 records across 36 lines, about 21 each) sit "
    "almost entirely below it, and reporting only pooled metrics would have hidden that.",
    "then jumps to 0.61 at 20-59 records and 0.70 beyond 60. One nuance cuts the other way and "
    "belongs in the open: the naive forecasts degrade far less — in the thinnest bin (5-9 prior "
    "records, n = 21) persistence still scores 0.58 and the rolling mean 0.47 while XGBoost "
    "collapses — so roughly a month of per-line history is the price of admission for the learned "
    "models specifically, and a newly commissioned line is best served by persistence until that "
    "history accumulates. Factory C’s lines (767 records across 36 lines, about 21 each) sit almost "
    "entirely below the threshold, and reporting only pooled metrics would have hidden that.")

# --- VI-J --------------------------------------------------------------
rep("Persistence must be the contractual baseline whenever forecasting tools are evaluated,",
    "Persistence and the per-line rolling mean must be the contractual baselines whenever "
    "forecasting tools are evaluated,")

# --- Limitations -------------------------------------------------------
rep("and the final test slice spans just ten days in early January",
    "and the final test slice spans just nine production dates in early January")
rep("Seed-paired tests, even Holm-corrected, speak only to initialization variance on one split, "
    "which is why the bootstrap intervals carry the generalization claims; both rest on a single "
    "quarter of data.",
    "Seed-paired tests, even Holm-corrected, speak only to initialization variance on one split and "
    "are underpowered at five seeds, which is why the line-clustered bootstrap intervals carry the "
    "generalization claims; those intervals respect within-line correlation but cannot respect "
    "within-date correlation (nine test dates is too few to resample), and both instruments rest on "
    "a single quarter of data.")

# --- Conclusion --------------------------------------------------------
rep("five seeds each, Holm-corrected paired tests, and naive plus persistence baselines.",
    "five seeds each, Holm-corrected paired tests, and naive, persistence, and rolling-mean baselines.")
rep("(LightGBM 0.641 ± 0.005 and XGBoost 0.631 ± 0.005 against 0.545 ± 0.035 average R², bootstrap "
    "CIs excluding zero)",
    "(LightGBM 0.641 ± 0.005 and XGBoost 0.631 ± 0.005 against 0.545 ± 0.035 average R², "
    "line-clustered bootstrap CIs excluding zero)")
rep("a ridge regression was statistically indistinguishable from the best network while posting the "
    "best defect-rate point accuracy in the study; and the single-task ablations",
    "a ridge regression was statistically indistinguishable from the best network while posting the "
    "best defect-rate point accuracy in the study; the strongest network cleared a raw rolling-mean "
    "passthrough of its own input features by a real but modest margin; and the single-task ablations")
rep("report seed variance with corrected tests, publish persistence and simple-model comparisons,",
    "report seed variance with corrected tests, cluster the bootstrap by the unit that actually "
    "repeats, publish persistence, rolling-mean, and simple-model comparisons,")

# --- Data availability -------------------------------------------------
rep("is provided to reviewers as supplementary material, and a public repository with an archival "
    "DOI will be linked here upon acceptance.",
    "is provided to reviewers as supplementary material, and a public repository with an archival "
    "DOI will be linked here upon acceptance. All supplementary artifacts have been re-audited so "
    "that factory and buyer identities appear nowhere outside the access-controlled anonymization "
    "key, which is excluded from the release.")

for old, new in R:
    k = s.count(old)
    assert k == 1, f'match count {k} for: {old[:70]}...'
    s = s.replace(old, new)

with io.open(DST, 'w', encoding='utf-8') as f:
    f.write(s)
print('wrote', DST, len(s), 'chars,', len(R), 'replacements applied')
