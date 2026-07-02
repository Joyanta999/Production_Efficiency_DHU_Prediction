#!/usr/bin/env python3
"""
Multi-task neural architectures with LEARNED CATEGORICAL EMBEDDINGS.

Change vs v5: the neural side no longer receives ordinal-coded categoricals as
plain numbers. Each categorical field (factory, building, buyer) gets its own
embedding table; the embeddings are concatenated with the scaled numeric
features and fed to the backbone. This removes the ordinal-encoding handicap
the v5 paper flagged as a limitation.

Backbones implemented (all multi-task: shared encoder -> 2 regression heads):
  MLP, DeepMLP, CNN1D, TabTransformer, BiLSTM      (the v5 five, now embedded)
  FTTransformer, ResNet                            (RTDL-style strong baselines)

Loss:
  fixed 0.5/0.5 weighting (default), OR
  Kendall-et-al. homoscedastic uncertainty weighting (learn log-variance per
  task) as an MTL ablation -- pass mtl="uncertainty".

Reference for the strong tabular baselines and uncertainty weighting:
  Gorishniy et al., "Revisiting Deep Learning Models for Tabular Data",
    NeurIPS 2021 (FT-Transformer, ResNet) -- "RTDL".
  Kendall, Gal & Cipolla, "Multi-Task Learning Using Uncertainty to Weigh
    Losses", CVPR 2018.
"""
import math
import numpy as np
import torch
import torch.nn as nn

import config as C


def emb_dim(cardinality):
    """Rule-of-thumb embedding width, capped."""
    return int(min(50, round(1.6 * cardinality ** 0.56)))


# --------------------------------------------------------------- input encoder
class TabularInput(nn.Module):
    """Numeric passthrough + per-field categorical embeddings -> flat vector."""
    def __init__(self, n_num, cat_cardinalities):
        super().__init__()
        self.n_num = n_num
        self.embs = nn.ModuleList([nn.Embedding(card, emb_dim(card))
                                   for card in cat_cardinalities])
        self.out_dim = n_num + sum(emb_dim(c) for c in cat_cardinalities)

    def forward(self, xnum, xcat):
        parts = [xnum]
        for i, emb in enumerate(self.embs):
            parts.append(emb(xcat[:, i]))
        return torch.cat(parts, dim=1)


# --------------------------------------------------------------- heads + loss
class TwoHeads(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        def head():
            return nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        self.eff = head()
        self.dhu = head()

    def forward(self, z):
        return torch.cat([self.eff(z), self.dhu(z)], dim=1)


class MTLLoss(nn.Module):
    """fixed weights or learned uncertainty weighting (Kendall et al.)."""
    def __init__(self, mode="fixed", w=(0.5, 0.5)):
        super().__init__()
        self.mode = mode
        if mode == "uncertainty":
            self.log_var = nn.Parameter(torch.zeros(2))
        else:
            self.register_buffer("w", torch.tensor(w, dtype=torch.float32))
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        le = self.mse(pred[:, 0], target[:, 0])
        ld = self.mse(pred[:, 1], target[:, 1])
        if self.mode == "uncertainty":
            pe = torch.exp(-self.log_var[0]); pdh = torch.exp(-self.log_var[1])
            return pe * le + self.log_var[0] + pdh * ld + self.log_var[1]
        return self.w[0] * le + self.w[1] * ld


# --------------------------------------------------------------- backbones
class _Base(nn.Module):
    """Wraps an encoder that maps the flat input vector -> latent, + 2 heads."""
    def __init__(self, n_num, cat_cards, latent, hidden_head=64):
        super().__init__()
        self.inp = TabularInput(n_num, cat_cards)
        self.encoder = self.build_encoder(self.inp.out_dim, latent)
        self.heads = TwoHeads(latent, hidden_head)

    def build_encoder(self, in_dim, latent):
        raise NotImplementedError

    def forward(self, xnum, xcat):
        z = self.encoder(self.inp(xnum, xcat))
        return self.heads(z)


class MLP(_Base):
    def __init__(self, n_num, cat_cards, h1=256, h2=128, **_):
        self._h = (h1, h2); super().__init__(n_num, cat_cards, latent=h2)
    def build_encoder(self, d, latent):
        h1, h2 = self._h
        return nn.Sequential(nn.Linear(d, h1), nn.ReLU(),
                             nn.Linear(h1, h2), nn.ReLU())


class DeepMLP(_Base):
    def __init__(self, n_num, cat_cards, dropout=0.3, **_):
        self._p = dropout; super().__init__(n_num, cat_cards, latent=64)
    def build_encoder(self, d, latent):
        p = self._p
        def blk(i, o):
            return [nn.Linear(i, o), nn.BatchNorm1d(o), nn.GELU(), nn.Dropout(p)]
        return nn.Sequential(*blk(d, 512), *blk(512, 256), *blk(256, 128), *blk(128, 64))


class CNN1D(_Base):
    def __init__(self, n_num, cat_cards, **_):
        super().__init__(n_num, cat_cards, latent=256)
    def build_encoder(self, d, latent):
        return _CNNEnc(d)


class _CNNEnc(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1))
    def forward(self, x):
        return self.net(x.unsqueeze(1)).squeeze(-1)


class TabTransformer(_Base):
    """Tokenises every feature (continuous included) into d_model tokens, runs
    Transformer encoder blocks, mean-pools. (Same departure from Huang et al.
    as v5: continuous fields are tokenised too.)"""
    def __init__(self, n_num, cat_cards, d_model=64, heads=4, blocks=3, ff=256, **_):
        self._cfg = (d_model, heads, blocks, ff)
        super().__init__(n_num, cat_cards, latent=d_model)
    def build_encoder(self, d, latent):
        return _TokenTransformer(d, *self._cfg)


class _TokenTransformer(nn.Module):
    def __init__(self, n_features, d_model, heads, blocks, ff):
        super().__init__()
        self.proj = nn.Linear(1, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_features, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, batch_first=True,
                                           dropout=0.1, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, blocks)
    def forward(self, x):                       # x: (B, F)
        tok = self.proj(x.unsqueeze(-1)) + self.pos
        return self.enc(tok).mean(dim=1)


class BiLSTM(nn.Module):
    """Per-line sliding window of length seq_len over the flat embedded input.
    The window is built in train.py (sequence tensors); here we just consume it.
    Falls back to treating the single record as a length-1 sequence."""
    def __init__(self, n_num, cat_cards, hidden=128, layers=2, seq_len=7, **_):
        super().__init__()
        self.inp = TabularInput(n_num, cat_cards)
        self.lstm = nn.LSTM(self.inp.out_dim, hidden, layers,
                            batch_first=True, bidirectional=True, dropout=0.1)
        self.norm = nn.LayerNorm(2 * hidden)
        self.drop = nn.Dropout(0.1)
        self.heads = TwoHeads(2 * hidden)
        self.seq_len = seq_len
    def forward(self, xnum, xcat):
        # xnum: (B, T, n_num), xcat: (B, T, n_cat)  -- sequence inputs
        if xnum.dim() == 2:
            xnum = xnum.unsqueeze(1); xcat = xcat.unsqueeze(1)
        B, T, _ = xnum.shape
        flat = self.inp(xnum.reshape(B * T, -1), xcat.reshape(B * T, -1))
        seq = flat.reshape(B, T, -1)
        out, _ = self.lstm(seq)
        z = self.drop(self.norm(out[:, -1, :]))
        return self.heads(z)


# ----------------------------------------------------- RTDL-style strong models
class ResNet(_Base):
    """RTDL ResNet: stack of pre-norm residual blocks over the embedded input."""
    def __init__(self, n_num, cat_cards, d=256, blocks=3, dropout=0.2, **_):
        self._cfg = (d, blocks, dropout); super().__init__(n_num, cat_cards, latent=d)
    def build_encoder(self, in_dim, latent):
        return _ResNetEnc(in_dim, *self._cfg)


class _ResNetEnc(nn.Module):
    def __init__(self, in_dim, d, blocks, dropout):
        super().__init__()
        self.first = nn.Linear(in_dim, d)
        self.blocks = nn.ModuleList([self._block(d, dropout) for _ in range(blocks)])
        self.last_norm = nn.BatchNorm1d(d)
    def _block(self, d, p):
        return nn.ModuleDict(dict(
            norm=nn.BatchNorm1d(d),
            lin1=nn.Linear(d, d * 2), lin2=nn.Linear(d * 2, d),
            drop=nn.Dropout(p)))
    def forward(self, x):
        x = self.first(x)
        for b in self.blocks:
            h = b["norm"](x); h = torch.relu(b["lin1"](h))
            h = b["drop"](h); h = b["lin2"](h)
            x = x + h
        return torch.relu(self.last_norm(x))


class FTTransformer(nn.Module):
    """RTDL FT-Transformer: numeric features are linearly tokenised, categoricals
    use embedding tokens, a [CLS] token is prepended, Transformer blocks run,
    and the CLS representation feeds the two heads."""
    def __init__(self, n_num, cat_cards, d_model=64, heads=8, blocks=3, ff=128,
                 dropout=0.1, **_):
        super().__init__()
        self.n_num = n_num
        self.num_tok = nn.Parameter(torch.randn(n_num, d_model) * 0.02)
        self.num_bias = nn.Parameter(torch.zeros(n_num, d_model))
        self.cat_emb = nn.ModuleList([nn.Embedding(card, d_model) for card in cat_cards])
        self.cls = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, ff, batch_first=True,
                                           dropout=dropout, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, blocks)
        self.norm = nn.LayerNorm(d_model)
        self.heads = TwoHeads(d_model)
    def forward(self, xnum, xcat):
        B = xnum.shape[0]
        num_tokens = xnum.unsqueeze(-1) * self.num_tok + self.num_bias    # (B,n_num,d)
        cat_tokens = [emb(xcat[:, i]).unsqueeze(1) for i, emb in enumerate(self.cat_emb)]
        cls = self.cls.expand(B, -1, -1)
        tokens = torch.cat([cls, num_tokens] + cat_tokens, dim=1)
        z = self.enc(tokens)[:, 0, :]            # CLS
        return self.heads(self.norm(z))


# --------------------------------------------------------------- registry
SEQUENCE_MODELS = {"BiLSTM"}          # need per-line windows
MODEL_ZOO = {
    "MLP": MLP, "DeepMLP": DeepMLP, "CNN1D": CNN1D,
    "TabTransformer": TabTransformer, "BiLSTM": BiLSTM,
    "FTTransformer": FTTransformer, "ResNet": ResNet,
}


def build_model(name, n_num, cat_cards, **kw):
    return MODEL_ZOO[name](n_num=n_num, cat_cards=cat_cards, **kw)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
