"""Expert demonstrations and lock-step rollouts (no PyTorch; shared by train.py and run.py)."""
from __future__ import annotations

import numpy as np

from features import CueState, TextState, action_index, encode_cue, encode_text, text_target
from pocketos import SPLITS, expert_action, make_task, optimal_steps, split_seeds, task_broken, task_done


# ------------------------------------------------------------ data ----
def _expert_episode(seed):
    phone, task = make_task(seed, **SPLITS["train"])
    cs, ts = CueState(task.prompt), TextState(task.prompt)
    cue, text = [], []
    while not task_done(phone, task):
        verb, el = expert_action(phone, task)
        enc, els = encode_cue(phone, cs)
        cue.append((enc, action_index(enc, els, verb, el)))
        ids, _ = encode_text(phone, ts)
        text.append((np.array(ids, np.uint8), np.array(text_target(verb, el), np.uint8)))
        cs.update(phone.cur, verb, el)
        ts.update(phone.cur, verb, el)
        phone.step(verb, el)
    return cue, text


def make_demos(n_episodes, offset=0, chunk=1000):
    """Yield (cue_samples, text_samples) for chunks of expert episodes. Single process, low RAM."""
    seeds = list(split_seeds("train", n_episodes, offset))
    for i in range(0, len(seeds), chunk):
        eps = [_expert_episode(sd) for sd in seeds[i:i + chunk]]
        yield [x for c, _ in eps for x in c], [x for _, t in eps for x in t]


class Single:
    """Wrap a per-env policy (baselines) in the batched interface."""

    def __init__(self, make):
        self.make = make

    def start(self, task):
        return self.make()

    def act(self, phones, states):
        return [s(p, s.task_ref) for p, s in zip(phones, states)]


def rollout(policy, seeds, split, max_factor=2.0, collect=False, **overrides):
    """Run many episodes in lockstep. Returns success list (+ expert-labelled states if collect)."""
    kw = {**SPLITS[split], **overrides}
    envs = []
    for s in seeds:
        budget = int(max_factor * optimal_steps(s, **kw))
        phone, task = make_task(s, **kw)
        st = policy.start(task)
        if isinstance(policy, Single):
            st.task_ref = task
        envs.append(dict(phone=phone, task=task, st=st, left=budget, ok=False, done=False,
                         cs=CueState(task.prompt), ts=TextState(task.prompt)))
    labelled_cue, labelled_text = [], []
    while True:
        act = [e for e in envs if not e["done"]]
        if not act:
            break
        actions = policy.act([e["phone"] for e in act], [e["st"] for e in act])
        for e, (verb, el) in zip(act, actions):
            p = e["phone"]
            if collect:  # DAgger: label the state the learner reached with the expert's action
                xv, xe = expert_action(p, e["task"])
                enc, els = encode_cue(p, e["cs"])
                labelled_cue.append((enc, action_index(enc, els, xv, xe)))
                ids, _ = encode_text(p, e["ts"])
                labelled_text.append((np.array(ids, np.uint8), np.array(text_target(xv, xe), np.uint8)))
            # agent memory follows the agent's own actions
            if verb != "NOOP":
                if isinstance(e["st"], (CueState, TextState)):
                    e["st"].update(p.cur, verb, el)
                e["cs"].update(p.cur, verb, el)
                e["ts"].update(p.cur, verb, el)
            p.step(verb, el)
            e["left"] -= 1
            if task_done(p, e["task"]):
                e["ok"], e["done"] = True, True
            elif e["left"] <= 0 or task_broken(p, e["task"]):
                e["done"] = True
    succ = [e["ok"] for e in envs]
    return (succ, labelled_cue, labelled_text) if collect else succ
