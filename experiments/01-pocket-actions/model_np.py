"""NumPy inference for the cue-and-pointer policy (no PyTorch needed).

Checkpoints are .npz files written by train.py: every tensor of the PyTorch
state_dict under its own name, plus a JSON "config" entry. The forward pass
here mirrors model_torch.CuePolicy exactly (tested in test_models.py).
"""
from __future__ import annotations

import json
import math

import numpy as np

from features import CueState, collate, decode_action, encode_cue, unpad_index

GELU_C = math.sqrt(2 / math.pi)  # Python float: keeps arrays float32 under NumPy 2 promotion


def gelu(x):
    return 0.5 * x * (1 + np.tanh(GELU_C * (x + 0.044715 * x * x * x)))


def layer_norm(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


def linear(x, P, name):
    wt = P.get(name + ".weight.T")
    y = x @ (wt if wt is not None else P[name + ".weight"].T)
    b = P.get(name + ".bias")
    return y if b is None else y + b


def load(path):
    z = np.load(path, allow_pickle=False)
    cfg = json.loads(str(z["config"]))
    P = {k: z[k].astype(np.float32) for k in z.files if k != "config"}
    return cfg, P


def prepare(P):
    """Add contiguous transposed copies of linear weights (faster CPU matmuls)."""
    out = dict(P)
    for k, w in P.items():
        if w.ndim == 2 and not k.endswith((".T",)) and k.split(".")[0] not in (
                "hash", "role", "clause", "tick", "kind", "ftype", "prev", "ttype"):
            out[k + ".T"] = np.ascontiguousarray(w.T)
    return out


def quantize_int8(P):
    """Symmetric per-row int8 for every 2-D weight; returns dequantized copy and bytes."""
    Q, nbytes = {}, 0
    for k, w in P.items():
        if w.ndim == 2:
            s = np.abs(w).max(1, keepdims=True) / 127 + 1e-12
            q = np.clip(np.round(w / s), -127, 127)
            Q[k] = (q * s).astype(np.float32)
            nbytes += w.size + 4 * w.shape[0]
        else:
            Q[k] = w
            nbytes += 4 * w.size
    return Q, nbytes


def attention(x, pad, P, name, heads, causal=False):
    B, T, d = x.shape
    qkv = linear(x, P, name + ".qkv").reshape(B, T, 3, heads, d // heads).transpose(2, 0, 3, 1, 4)
    q, k, v = qkv[0], qkv[1], qkv[2]
    att = q @ k.transpose(0, 1, 3, 2) * (1.0 / math.sqrt(d // heads))
    if pad is not None:
        att = np.where(pad[:, None, None, :], -1e9, att)
    if causal:
        att = att + np.triu(np.full((T, T), -1e9, np.float32), 1)
    att = np.exp(att - att.max(-1, keepdims=True))
    att /= att.sum(-1, keepdims=True)
    y = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, d)
    return linear(y, P, name + ".out")


def block(x, pad, P, name, heads, causal=False):
    x = x + attention(layer_norm(x, P[name + ".ln1.weight"], P[name + ".ln1.bias"]), pad, P, name + ".attn", heads, causal)
    h = gelu(linear(layer_norm(x, P[name + ".ln2.weight"], P[name + ".ln2.bias"]), P, name + ".mlp.0"))
    return x + linear(h, P, name + ".mlp.2")


def cue_forward(cfg, P, b, no_match=False):
    """Logits [B, T, n_verbs] with invalid actions set to -inf."""
    E = lambda n, i: P[n + ".weight"][i]  # noqa: E731
    ctx = E("role", b["cue_role"]) + E("clause", b["cue_clause"]) + E("tick", b["cue_tick"])

    def agg(mask):  # mask [..., Lc] -> mean of cue context vectors
        s = mask.sum(-1, keepdims=True)
        return (mask @ ctx if mask.ndim == 3 else np.einsum("bl,bld->bd", mask, ctx)) / np.maximum(s, 1)

    match, pm, cm = b["match"], b["path_match"], b["clip_match"]
    el_feat = b["el_feat"]
    if no_match or cfg.get("no_match"):
        match, pm, cm = np.zeros_like(match), np.zeros_like(pm), np.zeros_like(cm)
        el_feat = el_feat.copy()
        el_feat[..., 7:9] = 0
    st = (linear(b["st_feat"], P, "st_proj") + E("prev", b["prev"]) + E("ttype", 0)
          + linear(agg(pm), P, "a_path") + linear(agg(cm), P, "a_clip"))
    cue = E("hash", b["cue_id"]) + ctx + E("ttype", 1)
    el = (E("kind", b["el_kind"]) + E("ftype", b["el_ftype"]) + linear(el_feat, P, "el_proj")
          + linear(agg(match), P, "a_el") + E("ttype", 2))
    x = np.concatenate([st[:, None], cue, el], 1).astype(np.float32)
    for _ in range(cfg["loops"]):
        for i in range(cfg["layers"]):
            x = block(x, b["pad"], P, f"blocks.{i}", cfg["heads"])
    x = layer_norm(x, P["ln_f.weight"], P["ln_f.bias"])
    logits = linear(x, P, "head")
    return np.where(b["valid"], logits, -np.inf)


class CueAgent:
    """Policy callable for pocketos.run_episode(policy, ...)."""

    def __init__(self, cfg, P, no_match=False, prompt_for=None):
        self.cfg, self.P, self.no_match = cfg, prepare(P), no_match
        self.prompt_for = prompt_for  # optional fn(task) -> prompt (negative control)
        self.task = None

    def __call__(self, phone, task):
        if task is not self.task:
            self.task = task
            self.cs = CueState(self.prompt_for(task) if self.prompt_for else task.prompt)
        enc, els = encode_cue(phone, self.cs)
        b = collate([enc])
        logits = cue_forward(self.cfg, self.P, b, self.no_match)[0].reshape(-1)
        Lc = b["cue_id"].shape[1]
        verb, el = decode_action(enc, els, unpad_index(enc, int(np.argmax(logits)), Lc))
        self.cs.update(phone.cur, verb, el)
        return verb, el


def n_params(P):
    return int(sum(v.size for k, v in P.items() if not k.endswith(".T")))


def random_params(cfg, seed=0):
    """Randomly initialised parameters with the same names/shapes as model_torch.CuePolicy."""
    from features import H, MAX_CLAUSE, N_EL_FEATS, N_ST_FEATS
    from pocketos import FTYPES, KINDS, VERBS
    rng = np.random.default_rng(seed)
    d, nv = cfg["d"], len(VERBS)
    shapes = {"hash.weight": (H, d), "role.weight": (3, d), "clause.weight": (MAX_CLAUSE, d),
              "tick.weight": (2, d), "kind.weight": (len(KINDS), d), "ftype.weight": (len(FTYPES) + 1, d),
              "prev.weight": (nv + 1, d), "ttype.weight": (3, d),
              "st_proj.weight": (d, N_ST_FEATS), "st_proj.bias": (d,),
              "el_proj.weight": (d, N_EL_FEATS), "el_proj.bias": (d,),
              "a_path.weight": (d, d), "a_clip.weight": (d, d), "a_el.weight": (d, d),
              "ln_f.weight": (d,), "ln_f.bias": (d,), "head.weight": (nv, d), "head.bias": (nv,)}
    for i in range(cfg["layers"]):
        b = f"blocks.{i}."
        shapes.update({b + "ln1.weight": (d,), b + "ln1.bias": (d,), b + "ln2.weight": (d,), b + "ln2.bias": (d,),
                       b + "attn.qkv.weight": (3 * d, d), b + "attn.qkv.bias": (3 * d,),
                       b + "attn.out.weight": (d, d), b + "attn.out.bias": (d,),
                       b + "mlp.0.weight": (4 * d, d), b + "mlp.0.bias": (4 * d,),
                       b + "mlp.2.weight": (d, 4 * d), b + "mlp.2.bias": (d,)})
    P = {}
    for k, s in shapes.items():
        if k.endswith(("ln1.weight", "ln2.weight", "ln_f.weight")):
            P[k] = np.ones(s, np.float32)
        elif k.endswith("bias"):
            P[k] = np.zeros(s, np.float32)
        else:
            P[k] = (rng.standard_normal(s) * 0.05).astype(np.float32)
    return P
