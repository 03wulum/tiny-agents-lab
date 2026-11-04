"""PyTorch models used for GPU training. Imported only by train.py and tests.

CuePolicy: cue-and-pointer policy (one decision per step, no text output).
TextGPT:   text-token baseline (reads the state as ASCII, writes "VERB label").
"""
from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from features import H, MAX_CLAUSE, N_EL_FEATS, N_ST_FEATS, TEXT_CTX, TEXT_VOCAB
from pocketos import FTYPES, KINDS, VERBS

NV = len(VERBS)


class Attn(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)

    def forward(self, x, pad=None, causal=False):
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, d // self.h).permute(2, 0, 3, 1, 4)
        mask = None
        if pad is not None:
            mask = ~pad[:, None, None, :]
            if causal:
                tri = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
                mask = (mask & tri) | torch.eye(T, dtype=torch.bool, device=x.device)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal and mask is None)
        return self.out(y.transpose(1, 2).reshape(B, T, d))


class Block(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attn(d, heads)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))

    def forward(self, x, pad=None, causal=False):
        x = x + self.attn(self.ln1(x), pad, causal)
        return x + self.mlp(self.ln2(x))


class CuePolicy(nn.Module):
    def __init__(self, d, layers, heads, loops=1, no_match=False):
        super().__init__()
        self.cfg = dict(kind="cue", d=d, layers=layers, heads=heads, loops=loops, no_match=no_match)
        self.hash = nn.Embedding(H, d)
        self.role, self.clause, self.tick = nn.Embedding(3, d), nn.Embedding(MAX_CLAUSE, d), nn.Embedding(2, d)
        self.kind, self.ftype = nn.Embedding(len(KINDS), d), nn.Embedding(len(FTYPES) + 1, d)
        self.prev, self.ttype = nn.Embedding(NV + 1, d), nn.Embedding(3, d)
        self.st_proj, self.el_proj = nn.Linear(N_ST_FEATS, d), nn.Linear(N_EL_FEATS, d)
        self.a_path, self.a_clip, self.a_el = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, NV)
        for e in (self.hash, self.role, self.clause, self.tick, self.kind, self.ftype, self.prev, self.ttype):
            nn.init.normal_(e.weight, std=0.02)

    def forward(self, b):
        ctx = self.role(b["cue_role"]) + self.clause(b["cue_clause"]) + self.tick(b["cue_tick"])
        match, pm, cm, ef = b["match"], b["path_match"], b["clip_match"], b["el_feat"]
        if self.cfg["no_match"]:
            match, pm, cm = torch.zeros_like(match), torch.zeros_like(pm), torch.zeros_like(cm)
            ef = ef.clone()
            ef[..., 7:9] = 0

        def agg(mask):
            s = mask.sum(-1, keepdim=True).clamp(min=1)
            out = mask @ ctx if mask.dim() == 3 else torch.einsum("bl,bld->bd", mask, ctx)
            return out / s

        z = torch.zeros_like(b["prev"])
        st = (self.st_proj(b["st_feat"]) + self.prev(b["prev"]) + self.ttype(z)
              + self.a_path(agg(pm)) + self.a_clip(agg(cm)))
        cue = self.hash(b["cue_id"]) + ctx + self.ttype(torch.ones_like(b["cue_id"]))
        el = (self.kind(b["el_kind"]) + self.ftype(b["el_ftype"]) + self.el_proj(ef)
              + self.a_el(agg(match)) + self.ttype(torch.full_like(b["el_kind"], 2)))
        x = torch.cat([st[:, None], cue, el], 1)
        for _ in range(self.cfg["loops"]):
            for blk in self.blocks:
                x = blk(x, b["pad"])
        logits = self.head(self.ln_f(x))
        return logits.masked_fill(~b["valid"], float("-inf"))


class TextGPT(nn.Module):
    def __init__(self, d, layers, heads):
        super().__init__()
        self.cfg = dict(kind="text", d=d, layers=layers, heads=heads)
        self.tok, self.pos = nn.Embedding(TEXT_VOCAB, d), nn.Embedding(TEXT_CTX, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, TEXT_VOCAB)
        nn.init.normal_(self.tok.weight, std=0.02)
        nn.init.normal_(self.pos.weight, std=0.02)

    def forward(self, ids, pad=None):
        """ids [B, T], left-padded; pad [B, T] True where padding."""
        if pad is None:
            pad = torch.zeros_like(ids, dtype=torch.bool)
        pos = ((~pad).long().cumsum(1) - 1).clamp(min=0)
        x = self.tok(ids) + self.pos(pos)
        for blk in self.blocks:
            x = blk(x, pad, causal=True)
        return self.head(self.ln_f(x))


def to_torch(batch, device):
    return {k: torch.as_tensor(v, device=device) for k, v in batch.items()}


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def save_npz(model, path):
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    np.savez(path, config=np.array(json.dumps(model.cfg)), **sd)


def cue_size(name):
    """Named sizes: (d, layers, heads, loops)."""
    return {"20k": (16, 2, 2, 1), "100k": (40, 3, 2, 1), "500k": (96, 4, 4, 1),
            "2m": (192, 4, 4, 1), "5m": (256, 6, 4, 1), "rec2m": (192, 4, 4, 4)}[name]


def text_size(name):
    return {"500k": (96, 4, 4), "2m": (192, 4, 4)}[name]
