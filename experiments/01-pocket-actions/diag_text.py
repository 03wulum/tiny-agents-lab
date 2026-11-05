"""Diagnose the text-token baseline (GPU, evaluation only, no training).

    python diag_text.py

For expert-visited states of the in-distribution split, it measures:
  * teacher-forced loss and per-character accuracy on the target action text
    (the same code path as training), and
  * greedy-decoded single-step accuracy (the same code path as evaluation),
    with a sample of raw decoded strings next to the expert's action.
If the teacher-forced accuracy is high but greedy decoding is near 0, the
decoder or parser has a bug. If both are low, the model is undertrained.
Writes results/diag_text.json.
"""
from __future__ import annotations

import json

import numpy as np
import torch

import model_np
import model_torch as MT
from features import TextState, encode_text, text_target
from pocketos import SPLITS, expert_action, make_task, split_seeds, task_done
from train import CKPT, RESULTS, TextPack, BatchedText


def expert_states(n_tasks):
    """Lists of (phone snapshot via replay, TextState, expert action) along expert trajectories."""
    out = []
    for s in split_seeds("id", n_tasks):
        phone, task = make_task(s, **SPLITS["id"])
        ts = TextState(task.prompt)
        step = 0
        while not task_done(phone, task):
            out.append((s, step))
            v, e = expert_action(phone, task)
            ts.update(phone.cur, v, e)
            phone.step(v, e)
            step += 1
    return out


def replay(seed, k):
    phone, task = make_task(seed, **SPLITS["id"])
    ts = TextState(task.prompt)
    for _ in range(k):
        v, e = expert_action(phone, task)
        ts.update(phone.cur, v, e)
        phone.step(v, e)
    return phone, task, ts


@torch.no_grad()
def main():
    device = "cuda"
    report = {}
    states = expert_states(60)
    for path in sorted(CKPT.glob("text-*.npz")):
        cfg, P = model_np.load(path)
        m = MT.TextGPT(cfg["d"], cfg["layers"], cfg["heads"])
        m.load_state_dict({k: torch.tensor(v) for k, v in P.items()})
        m = m.to(device).eval()
        # teacher-forced, training code path
        pack = TextPack(device)
        samples, envs = [], []
        for seed, k in states:
            phone, task, ts = replay(seed, k)
            v, e = expert_action(phone, task)
            ids, _ = encode_text(phone, ts)
            samples.append((np.array(ids, np.uint8), np.array(text_target(v, e), np.uint8)))
            envs.append((phone, ts, (v, e)))
        pack.add(samples)
        n = len(pack)
        idx = torch.arange(n, device=device)
        accs, exact, losses = [], [], []
        for i in range(0, n, 64):
            ids, pad, tgt = pack.batch(idx[i:i + 64])
            logits = m(ids, pad).float()
            mask = tgt != -100
            losses.append(torch.nn.functional.cross_entropy(logits[mask], tgt[mask]).item())
            hit = (logits.argmax(-1) == tgt) | ~mask
            accs.append(((logits.argmax(-1) == tgt) & mask).sum().item() / mask.sum().item())
            exact += hit.all(1).tolist()
        # greedy decoding, evaluation code path
        pol = BatchedText(m, device)
        acts = pol.act([p for p, _, _ in envs], [t for _, t, _ in envs])
        match = [a == x for a, (_, _, x) in zip(acts, envs)]
        noop = [a[0] == "NOOP" for a in acts]
        examples = [{"expert": f"{x[0]} {x[1].label if x[1] is not None else ''}".strip(), "decoded": r}
                    for (_, _, x), r in list(zip(envs, pol.last_raw))[:15]]
        report[path.stem] = dict(states=n, teacher_forced_loss=float(np.mean(losses)),
                                 teacher_forced_char_acc=float(np.mean(accs)),
                                 teacher_forced_exact=float(np.mean(exact)),
                                 greedy_step_acc=float(np.mean(match)), greedy_noop_rate=float(np.mean(noop)),
                                 examples=examples)
        print(path.stem, {k: v for k, v in report[path.stem].items() if k != "examples"})
        for ex in examples[:8]:
            print("   expert:", ex["expert"], "| decoded:", ex["decoded"])
    (RESULTS / "diag_text.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
