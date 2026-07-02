#!/usr/bin/env python3
"""
Per-architecture hyperparameter search (validation-tuned), addressing the v5
limitation that hyperparameters were shared across architectures rather than
tuned per model.

A small random search per model: each candidate is trained on train, scored by
validation loss (single seed for speed), and the best config is returned. Use
the returned hp dicts when training the final seed ensemble and in rolling
origin, so the deep-vs-trees comparison gives the networks a fair shot.

Search spaces are deliberately compact (CPU budget); widen as resources allow.
"""
import numpy as np

import config as C
from train import train_model

SPACES = {
    "TabTransformer": dict(d_model=[32, 64, 96], heads=[4, 8],
                           blocks=[2, 3, 4], ff=[128, 256]),
    "FTTransformer": dict(d_model=[64, 96], heads=[4, 8],
                          blocks=[2, 3, 4], ff=[128, 256], dropout=[0.1, 0.2]),
    "ResNet": dict(d=[128, 256], blocks=[2, 3, 4], dropout=[0.1, 0.2, 0.3]),
    "MLP": dict(h1=[128, 256, 512], h2=[64, 128]),
    "DeepMLP": dict(dropout=[0.1, 0.2, 0.3]),
}


def sample(space, rng):
    return {k: rng.choice(v).item() if hasattr(rng.choice(v), "item") else rng.choice(v)
            for k, v in space.items()}


def search(name, splits, n_trials=8, seed=42, dl=None):
    """Random search; returns (best_hp, best_val, trials)."""
    space = SPACES.get(name)
    if not space:
        return {}, None, []
    rng = np.random.default_rng(seed)
    best = (None, np.inf); trials = []
    for t in range(n_trials):
        hp = {k: rng.choice(v) for k, v in space.items()}
        hp = {k: (int(v) if isinstance(v, (np.integer,)) else
                  float(v) if isinstance(v, (np.floating,)) else v)
              for k, v in hp.items()}
        try:
            out = train_model(name, splits, seed, hp=hp,
                              dl=dl or _short_dl(), verbose=False)
            v = out["best_val"]
        except Exception as e:
            v = np.inf
        trials.append(dict(hp=hp, val=float(v)))
        if v < best[1]:
            best = (hp, v)
    return best[0] or {}, float(best[1]), trials


def _short_dl():
    """Cheaper training budget for the search phase."""
    d = dict(C.DL); d["max_epochs"] = 60; d["patience"] = 12
    return d


def tune_all(splits, models=("TabTransformer", "FTTransformer", "ResNet"),
             n_trials=8):
    out = {}
    for m in models:
        hp, val, trials = search(m, splits, n_trials=n_trials)
        out[m] = dict(best_hp=hp, best_val=val, n_trials=len(trials))
        print(f"[tune] {m}: best_val={val:.4f} hp={hp}")
    return out
