#!/usr/bin/env python3
"""
Training loop for the neural models: AdamW, ReduceLROnPlateau, gradient
clipping, early stopping with best-validation checkpoint restore.

Supports:
  * standard (per-row) models and the sequence model (BiLSTM) via per-line
    sliding windows built here
  * fixed 0.5/0.5 MTL loss and Kendall uncertainty-weighted loss
  * single-task ablation (train one head only) via task_weights
Returns test-set predictions in ORIGINAL target units plus the val/test metric
history for the training-duration sweep.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import config as C
from models import build_model, MTLLoss, SEQUENCE_MODELS


def set_seed(s):
    np.random.seed(s); torch.manual_seed(s)
    torch.use_deterministic_algorithms(False)


def _line_windows(part, Xnum, Xcat, seq_len):
    """Build per-line windows ending at each record: (n, T, .) left-padded by
    repeating the line's earliest record. Windows never cross line boundaries."""
    line = part[C.LINE_ID].values
    n = len(line)
    num = np.zeros((n, seq_len, Xnum.shape[1]), dtype=np.float32)
    cat = np.zeros((n, seq_len, Xcat.shape[1]), dtype=np.int64)
    # indices per line in order of appearance (df already chronological)
    from collections import defaultdict
    pos = defaultdict(list)
    for i, l in enumerate(line):
        pos[l].append(i)
    for l, idxs in pos.items():
        for k, i in enumerate(idxs):
            window = idxs[max(0, k - seq_len + 1):k + 1]
            pad = seq_len - len(window)
            rows = [idxs[0]] * pad + window
            num[i] = Xnum[rows]; cat[i] = Xcat[rows]
    return num, cat


def _tensors(part_obj, seq, seq_len):
    if seq:
        num, cat = _line_windows(part_obj.part, part_obj.Xnum, part_obj.Xcat, seq_len)
    else:
        num, cat = part_obj.Xnum.astype(np.float32), part_obj.Xcat.astype(np.int64)
    return (torch.tensor(num, dtype=torch.float32),
            torch.tensor(cat, dtype=torch.long),
            torch.tensor(part_obj.ycols, dtype=torch.float32))


def train_model(name, splits, seed, hp=None, mtl="fixed",
                task_weights=(0.5, 0.5), dl=C.DL, verbose=False):
    """Train one model/seed. Returns dict with test preds (original units),
    val metric, and per-epoch history (for the duration sweep on the lead model).
    task_weights=(1,0) or (0,1) yields a single-task ablation."""
    hp = hp or {}
    set_seed(seed)
    seq = name in SEQUENCE_MODELS
    n_num = splits.train.Xnum.shape[1]
    cards = splits.cat_cardinalities

    model = build_model(name, n_num, cards, seq_len=dl["seq_len"], **hp)
    # single-task ablation -> fixed weights with one task zeroed
    mode = mtl if task_weights == (0.5, 0.5) else "fixed"
    loss_fn = MTLLoss(mode=mode, w=task_weights)
    params = list(model.parameters()) + list(loss_fn.parameters())
    opt = torch.optim.AdamW(params, lr=dl["lr"], weight_decay=dl["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, factor=dl["lr_factor"], patience=dl["lr_patience"], min_lr=dl["lr_min"])

    tr_n, tr_c, tr_y = _tensors(splits.train, seq, dl["seq_len"])
    vl_n, vl_c, vl_y = _tensors(splits.val, seq, dl["seq_len"])
    te_n, te_c, te_y = _tensors(splits.test, seq, dl["seq_len"])
    loader = DataLoader(TensorDataset(tr_n, tr_c, tr_y),
                        batch_size=dl["batch_size"], shuffle=True)

    tsc = splits.target_scaler
    yte_o = splits.test.yorig

    def predict(n_t, c_t):
        model.eval()
        with torch.no_grad():
            p = model(n_t, c_t).numpy()
        return tsc.inverse_transform(p)

    best_val = np.inf; best_state = None; bad = 0; history = []
    for epoch in range(dl["max_epochs"]):
        model.train()
        for bn, bc, by in loader:
            opt.zero_grad()
            out = model(bn, bc)
            loss = loss_fn(out, by)
            loss.backward()
            nn.utils.clip_grad_norm_(params, dl["grad_clip"])
            opt.step()
        # validation
        model.eval()
        with torch.no_grad():
            vpred = model(vl_n, vl_c)
            vloss = loss_fn(vpred, vl_y).item()
        sched.step(vloss)
        history.append(dict(epoch=epoch + 1, val_loss=vloss))
        if vloss < best_val - 1e-6:
            best_val = vloss; bad = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= dl["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    pred_te = predict(te_n, te_c)
    return dict(model=name, seed=seed,
                pred_eff=pred_te[:, 0], pred_dhu=pred_te[:, 1],
                true_eff=yte_o[:, 0], true_dhu=yte_o[:, 1],
                n_params=sum(p.numel() for p in model.parameters()),
                epochs=len(history), best_val=best_val, history=history)


def predict_val(name, splits, seed, hp=None, mtl="fixed", dl=C.DL):
    """Convenience: train and also return validation-set DHU predictions (for
    breach-alert cutoff tuning)."""
    out = train_model(name, splits, seed, hp=hp, mtl=mtl, dl=dl)
    return out
