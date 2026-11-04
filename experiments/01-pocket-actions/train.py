"""GPU training + evaluation for experiment 01. Runs on Fortran's NVIDIA GPU only.

    python train.py                 # full sweep, <= 15 min of training, then GPU evaluation
    python train.py --smoke         # 1-minute end-to-end check of the whole pipeline
    python train.py --eval-only     # re-evaluate saved checkpoints

Writes checkpoints/*.npz (kept models, <= 50 MB total) and results/eval.json.
run.py turns eval.json into the PASS/FAIL table; it never trains.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import model_torch as MT
from baselines import KeywordAgent
from features import (EOS, TEXT_OUT, CueState, TextState, collate, decode_action, encode_cue, encode_text,
                      padded_index, parse_text_action, unpad_index)
from pocketos import SPLITS, make_task, split_seeds
from rollout import Single, make_demos, rollout

HERE = Path(__file__).parent
CKPT = HERE / "checkpoints"
RESULTS = HERE / "results"

# name: (family, size, train seed, seconds of training, n demo episodes, extra)
RUNS = [
    ("cue-20k", "cue", "20k", 0, 20, 20000, {}),
    ("cue-100k", "cue", "100k", 0, 20, 20000, {}),
    ("cue-500k", "cue", "500k", 0, 30, 20000, {}),
    ("cue-2m-s0", "cue", "2m", 0, 45, 20000, {}),
    ("cue-2m-s1", "cue", "2m", 1, 45, 20000, {}),
    ("cue-2m-s2", "cue", "2m", 2, 45, 20000, {}),
    ("cue-5m", "cue", "5m", 0, 55, 20000, {}),
    ("cue-rec2m", "cue", "rec2m", 0, 45, 20000, {}),
    ("cue-2m-nomatch", "cue", "2m", 0, 30, 20000, {"no_match": True}),
    ("cue-2m-demos1k", "cue", "2m", 0, 20, 1000, {}),
    ("cue-2m-demos5k", "cue", "2m", 0, 20, 5000, {}),
    ("text-500k", "text", "500k", 0, 45, 20000, {}),
    ("text-2m-s0", "text", "2m", 0, 55, 20000, {}),
    ("text-2m-s1", "text", "2m", 1, 55, 20000, {}),
    ("text-2m-s2", "text", "2m", 2, 55, 20000, {}),
]
TRAIN_CAP_SECONDS = 14 * 60   # hard cap on total training wall time (SOP: <= 15 min)
DAGGER_ROUNDS = 3


# --------------------------------------------------------- batches ----
def cue_batch(samples, device):
    encs = [e for e, _ in samples]
    b = collate(encs)
    Lc = b["cue_id"].shape[1]
    y = torch.tensor([padded_index(e, a, Lc) for e, a in samples], device=device)
    return MT.to_torch(b, device), y


def text_batch(samples, device):
    seqs = [np.concatenate([x, y]) for x, y in samples]
    T = max(len(s) for s in seqs)
    ids = np.zeros((len(seqs), T), np.int64)
    pad = np.ones((len(seqs), T), bool)
    tgt = np.full((len(seqs), T), -100, np.int64)
    for i, ((x, y), s) in enumerate(zip(samples, seqs)):
        o = T - len(s)                       # left padding
        ids[i, o:] = s
        pad[i, o:] = False
        tgt[i, o + len(x) - 1:T - 1] = y     # predict y from the position before each token
    t = lambda a: torch.as_tensor(a, device=device)  # noqa: E731
    return t(ids), t(pad), t(tgt)


def loss_fn(model, family, samples, device):
    if family == "cue":
        b, y = cue_batch(samples, device)
        logits = model(b).flatten(1)
        return F.cross_entropy(logits, y)
    ids, pad, tgt = text_batch(samples, device)
    return F.cross_entropy(model(ids, pad).flatten(0, 1), tgt.flatten(), ignore_index=-100)


# ------------------------------------------------------- policies ----
class BatchedCue:
    def __init__(self, model, device, prompt_for=None):
        self.model, self.device, self.prompt_for = model, device, prompt_for

    def start(self, task):
        return CueState(self.prompt_for(task) if self.prompt_for else task.prompt)

    @torch.no_grad()
    def act(self, phones, states):
        encs = [encode_cue(p, s) for p, s in zip(phones, states)]
        b = collate([e for e, _ in encs])
        Lc = b["cue_id"].shape[1]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
            logits = self.model(MT.to_torch(b, self.device)).flatten(1).float()
        idx = logits.argmax(1).tolist()
        return [decode_action(e, els, unpad_index(e, i, Lc)) for (e, els), i in zip(encs, idx)]


class BatchedText:
    def __init__(self, model, device):
        self.model, self.device = model, device

    def start(self, task):
        return TextState(task.prompt)

    @torch.no_grad()
    def act(self, phones, states, chunk=64):
        if len(phones) > chunk:
            return [a for i in range(0, len(phones), chunk)
                    for a in self.act(phones[i:i + chunk], states[i:i + chunk], chunk)]
        prompts = [encode_text(p, s)[0] for p, s in zip(phones, states)]
        T = max(len(x) for x in prompts)
        ids = torch.zeros((len(prompts), T), dtype=torch.long, device=self.device)
        pad = torch.ones((len(prompts), T), dtype=torch.bool, device=self.device)
        for i, x in enumerate(prompts):
            ids[i, T - len(x):] = torch.tensor(x, device=self.device)
            pad[i, T - len(x):] = False
        out = []
        for _ in range(TEXT_OUT):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                nxt = self.model(ids, pad)[:, -1].float().argmax(-1)
            out.append(nxt)
            ids = torch.cat([ids, nxt[:, None]], 1)
            pad = torch.cat([pad, torch.zeros_like(pad[:, :1])], 1)
            if bool((torch.stack(out, 1) == EOS).any(1).all()):
                break
        out = torch.stack(out, 1).tolist()
        acts = []
        for p, o in zip(phones, out):
            o = o[:o.index(EOS)] if EOS in o else o
            acts.append(parse_text_action(p, o)[0])
        return acts


# -------------------------------------------------------- training ----
def build(family, size, extra):
    if family == "cue":
        d, L, h, loops = MT.cue_size(size)
        return MT.CuePolicy(d, L, h, loops, **extra)
    d, L, h = MT.text_size(size)
    return MT.TextGPT(d, L, h)


def train_run(name, family, size, seed, seconds, n_demos, extra, demos, device, log):
    torch.manual_seed(seed)
    random.seed(seed)
    model = build(family, size, extra).to(device)
    data = list((demos["cue"] if family == "cue" else demos["text"])[:n_demos])
    bs = 256 if family == "cue" else 64
    big = MT.n_params(model) > 1_000_000
    peak_lr = 5e-4 if big else 2e-3
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)
    phases = [0.55] + [0.45 / DAGGER_ROUNDS] * DAGGER_ROUNDS
    t0, step, rng = time.time(), 0, random.Random(seed)
    train_time = 0.0
    dagger_offset = 1_000_000 + 100_000 * seed
    policy_cls = BatchedCue if family == "cue" else BatchedText
    for ph, frac in enumerate(phases):
        if ph > 0:  # DAgger round: roll out the learner, label with the expert, aggregate
            model.eval()
            n_ep = 300 if family == "cue" else 120
            seeds = split_seeds("train", n_ep, dagger_offset + ph * 10_000)
            _, lc, lt = rollout(policy_cls(model, device), seeds, "train", collect=True)
            data += lc if family == "cue" else lt
            model.train()
        ts = time.time()
        while time.time() - ts < frac * seconds:
            elapsed = train_time + (time.time() - ts)
            lr = peak_lr * min(1.0, (step + 1) / 100) * max(0.05, 1 - elapsed / seconds)
            for g in opt.param_groups:
                g["lr"] = lr
            batch = [data[rng.randrange(len(data))] for _ in range(bs)]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = loss_fn(model, family, batch, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            step += 1
        train_time += time.time() - ts
        if ph == 0 and name == "cue-2m-s0":
            MT.save_npz(model, CKPT / "cue-2m-s0-bconly.npz")
        log(f"{name}: phase {ph} done, step {step}, loss {loss.item():.3f}, {train_time:.0f}s training")
    model.eval()
    path = CKPT / f"{name}.npz"
    MT.save_npz(model, path)
    return model, dict(params=MT.n_params(model), steps=step, train_seconds=round(train_time, 1),
                       wall_seconds=round(time.time() - t0, 1), bytes=path.stat().st_size,
                       examples=len(data))


# ------------------------------------------------------ evaluation ----
def load_cue(path, device):
    import model_np
    cfg, P = model_np.load(path)
    m = MT.CuePolicy(cfg["d"], cfg["layers"], cfg["heads"], cfg["loops"], cfg.get("no_match", False))
    m.load_state_dict({k: torch.tensor(v) for k, v in P.items()})
    return m.to(device).eval()


def int8_copy(model, device):
    import model_np
    P = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    Q, nbytes = model_np.quantize_int8(P)
    m = type(model)(**{k: v for k, v in model.cfg.items() if k != "kind"})
    m.load_state_dict({k: torch.tensor(v) for k, v in Q.items()})
    return m.to(device).eval(), nbytes


def evaluate(models, n, device, log):
    """models: {name: torch model}. Returns nested dict of success rates."""
    splits = ["id", "unseen_apps", "long", "wording"]
    res = {}

    def score(policy, split, **kw):
        s = rollout(policy, split_seeds(split, n), split, **kw)
        return round(float(np.mean(s)), 4)

    for name, m in models.items():
        pol = BatchedCue(m, device) if name.startswith("cue") else BatchedText(m, device)
        res[name] = {sp: score(pol, sp) for sp in splits}
        log(f"eval {name}: {res[name]}")
    # baselines (CPU, no learning)
    res["keyword"] = {sp: score(Single(lambda: KeywordAgent(0)), sp) for sp in splits}
    from baselines import RandomAgent
    res["random"] = {sp: score(Single(lambda: RandomAgent(0)), sp) for sp in splits}
    # controls on the headline model
    m = models["cue-2m-s0"]
    other = {}

    def wrong_prompt(task):  # negative control: another task's instruction
        if task.seed not in other:
            other[task.seed] = make_task(task.seed + 777_777, **SPLITS["id"])[1].prompt
        return other[task.seed]

    res["control_blind_prompt"] = {sp: score(BatchedCue(m, device, wrong_prompt), sp) for sp in ("id", "unseen_apps")}
    rnd = build("cue", "2m", {}).to(device).eval()
    res["control_random_weights"] = {sp: score(BatchedCue(rnd, device), sp) for sp in ("id", "unseen_apps")}
    q, qbytes = int8_copy(m, device)
    res["perturb_int8_cue-2m-s0"] = {sp: score(BatchedCue(q, device), sp) for sp in splits}
    res["perturb_int8_cue-2m-s0"]["bytes"] = qbytes
    res["perturb_clutter_cue-2m-s0"] = {"unseen_apps": score(BatchedCue(m, device), "unseen_apps", clutter=1.5)}
    res["perturb_clutter_keyword"] = {"unseen_apps": score(Single(lambda: KeywordAgent(0)), "unseen_apps", clutter=1.5)}
    bc = CKPT / "cue-2m-s0-bconly.npz"
    if bc.exists():
        res["perturb_bconly_cue-2m-s0"] = {sp: score(BatchedCue(load_cue(bc, device), device), sp) for sp in splits}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end run (~1-2 min)")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--n-eval", type=int, default=500)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found. Training runs on the NVIDIA GPU only (SOP v0.6).")
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True
    CKPT.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    logf = open(RESULTS / ("train_log_smoke.txt" if args.smoke else "train_log.txt"), "w")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    runs = RUNS
    n_eval = args.n_eval
    if args.smoke:
        runs = [r[:4] + (5, 300, r[6]) for r in RUNS if r[0] in ("cue-20k", "cue-2m-s0", "text-500k", "text-2m-s0")]
        n_eval = 20
    log(f"GPU: {torch.cuda.get_device_name(0)}, {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
    meta = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "smoke": args.smoke, "runs": {}}
    models = {}
    if not args.eval_only:
        t = time.time()
        n_demo = max(r[5] for r in runs)
        cue, text = make_demos(n_demo, procs=args.procs)
        meta["demo_seconds"] = round(time.time() - t, 1)
        log(f"generated {n_demo} expert episodes: {len(cue)} decisions in {meta['demo_seconds']}s (CPU, simulator only)")
        demos = {"cue": cue, "text": text}
        used, planned = 0.0, sum(r[4] for r in runs)
        for name, fam, size, seed, secs, nd, extra in runs:
            # n_demos counts episodes; convert to decisions (about 11 per episode)
            n_dec = len(cue) if nd >= n_demo else int(len(cue) * nd / n_demo)
            # keep total training wall time (gradient steps + DAgger rollouts) under the cap
            scale = min(1.0, max(0.1, (TRAIN_CAP_SECONDS - used) / (1.4 * planned)))
            planned -= secs
            m, info = train_run(name, fam, size, seed, secs * scale, n_dec, extra, demos, device, log)
            used += info["wall_seconds"]
            models[name] = m
            meta["runs"][name] = info
        # training time = wall time of every run, including DAgger rollouts (data generation excluded)
        meta["train_minutes_total"] = round(sum(r["wall_seconds"] for r in meta["runs"].values()) / 60, 2)
        log(f"total training time {meta['train_minutes_total']} min")
    else:
        prev = json.loads((RESULTS / "eval.json").read_text())
        meta.update({k: prev[k] for k in ("runs", "train_minutes_total", "demo_seconds") if k in prev})
        for p in sorted(CKPT.glob("*.npz")):
            if p.stem.startswith("cue") and not p.stem.endswith("bconly"):
                models[p.stem] = load_cue(p, device)
            elif p.stem.startswith("text"):
                import model_np
                cfg, P = model_np.load(p)
                m = MT.TextGPT(cfg["d"], cfg["layers"], cfg["heads"])
                m.load_state_dict({k: torch.tensor(v) for k, v in P.items()})
                models[p.stem] = m.to(device).eval()
    t = time.time()
    meta["success"] = evaluate(models, n_eval, device, log)
    meta["eval_seconds"] = round(time.time() - t, 1)
    meta["n_eval"] = n_eval
    meta["checkpoint_bytes"] = {p.name: p.stat().st_size for p in sorted(CKPT.glob("*.npz"))}
    out = RESULTS / ("eval_smoke.json" if args.smoke else "eval.json")
    out.write_text(json.dumps(meta, indent=1))
    log(f"wrote {out.name}; eval took {meta['eval_seconds']}s")


if __name__ == "__main__":
    main()
