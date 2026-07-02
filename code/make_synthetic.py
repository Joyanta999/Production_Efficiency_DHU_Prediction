#!/usr/bin/env python3
"""
Generate a schema-faithful SYNTHETIC daily-production export so the whole
pipeline can be exercised end-to-end without the proprietary VistaQ data.

It reproduces the column names, dtypes, per-line autocorrelation (so persistence
and rolling-mean baselines behave sensibly), the dirty values the cleaner is
meant to remove, and the date range. It does NOT reproduce the real factory's
numbers -- it is only for testing that the code runs and the tables populate.

Usage:
    python make_synthetic.py --rows 8000 --out ../data/VistaQDailyProduction.csv
"""
import argparse
import numpy as np
import pandas as pd


def generate(n_lines=140, days=85, seed=0):
    rng = np.random.default_rng(seed)
    factories = ["Alpha Apparels Ltd", "Beta Garments Ltd", "Gamma Textiles Ltd"]
    fac_for_line = rng.choice(factories, n_lines, p=[0.55, 0.30, 0.15])
    buildings = {f: [f"{f.split()[0]}-B{i}" for i in range(1, 4)] for f in factories}
    buyers = [f"Buyer {i}" for i in range(1, 24)]
    buyer_p = np.array([0.47, 0.15, 0.13] + [0.25 / 20] * 20)
    buyer_p /= buyer_p.sum()

    start = pd.Timestamp("2025-10-22")
    rows = []
    # per-line latent levels for autocorrelation
    eff_level = rng.normal(59, 18, n_lines).clip(20, 95)
    dhu_level = rng.lognormal(1.9, 0.5, n_lines).clip(1, 25)
    for li in range(n_lines):
        fac = fac_for_line[li]
        bld = rng.choice(buildings[fac])
        buyer = rng.choice(buyers, p=buyer_p)
        smv = rng.uniform(8, 45)
        eff_prev = eff_level[li]; dhu_prev = dhu_level[li]
        for d in range(days):
            if rng.random() < 0.25:               # idle / no record day
                continue
            date = start + pd.Timedelta(days=d)
            # AR(1)-ish dynamics
            eff = 0.7 * eff_prev + 0.3 * eff_level[li] + rng.normal(0, 8)
            dhu = 0.7 * dhu_prev + 0.3 * dhu_level[li] + rng.normal(0, 2)
            eff = float(np.clip(eff, 5, 120)); dhu = float(max(0.0, dhu))
            mp = max(1, int(rng.normal(45, 8)))
            row = dict(
                Date=date.strftime("%Y-%m-%d"),
                WorkspaceLineId=f"L{li:03d}",
                WorkspaceFactoryName=fac,
                WorkspaceBuildingName=bld,
                BuyerName=buyer,
                SMV=round(smv, 2), SampleSMV=round(smv * rng.uniform(0.9, 1.1), 2),
                CM=round(rng.uniform(0.5, 3.0), 2),
                DayTarget=int(rng.uniform(500, 3000)),
                IETarget=int(rng.uniform(500, 3000)),
                TargetEfficiency=round(rng.uniform(50, 80), 1),
                ManPowerPresent=mp,
                PlannedIronMan=2, PlannedHelper=8, PlannedOperator=30,  # constant
                PlannedHours=round(rng.uniform(8, 11), 1),
                RunningWorkDay=int(rng.uniform(1, 26)),
                AchievedEfficiency=round(eff, 2),
                dhu=round(dhu, 2),
            )
            rows.append(row); eff_prev, dhu_prev = eff, dhu

    df = pd.DataFrame(rows)
    # inject the dirty values the cleaner removes
    bad = df.sample(frac=0.05, random_state=seed).index
    df.loc[bad[: len(bad) // 2], "AchievedEfficiency"] = rng.uniform(200, 14000,
                                                                     len(bad) // 2)
    df.loc[bad[len(bad) // 2:], "dhu"] = rng.uniform(60, 130, len(bad) - len(bad) // 2)
    # a few genuine zero-DHU rows
    z = df.sample(frac=0.02, random_state=seed + 1).index
    df.loc[z, "dhu"] = 0.0
    return df.sort_values("Date").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8000)
    ap.add_argument("--out", default="../data/VistaQDailyProduction.csv")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import os
    df = generate(seed=args.seed)
    # trim/extend roughly to requested row count by adjusting nothing fancy
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"wrote {len(df)} synthetic rows -> {args.out}")
