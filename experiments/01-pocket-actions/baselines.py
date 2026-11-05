"""Non-learned baselines for PocketOS: random agent and keyword heuristic."""
from __future__ import annotations

import random
import re

VERB_WORDS = {
    "CUT": {"move", "put", "transfer", "relocate", "shift"},
    "DELETE": {"delete", "remove", "trash", "erase", "rid"},
    "SHARE": {"share", "send", "forward", "pass"},
}


def cues(prompt: str) -> set:
    """The instruction as an unordered bag of cue words (file names keep their extension)."""
    return set(re.findall(r"[a-z0-9.]+", prompt.lower())) | set(re.findall(r"[a-z]+", prompt.lower()))


def label_hits(label: str, cue: set) -> bool:
    return label in cue or all(w in cue for w in label.split())


class RandomAgent:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)

    def __call__(self, phone, task):
        return self.rng.choice(phone.valid_actions())


class KeywordAgent:
    """Greedy: act on whatever on-screen element matches a cue word; explore otherwise.

    It has no notion of sub-goal order, superlatives ("newest") or which
    folder is a destination vs a source, which is exactly what a learned
    policy has to pick up.
    """

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.task = None

    def _reset(self, task):
        self.task, self.cue = task, cues(task.prompt)
        self.done_uids, self.opened = set(), set()
        self.verbs = [v for v, ws in VERB_WORDS.items() if ws & self.cue] or ["CUT"]

    def __call__(self, phone, task):
        if task is not self.task:
            self._reset(task)
        screen = phone.screen()
        if phone.sharing is not None:
            hits = [e for e in screen if e.label in self.cue]
            return ("SEND", hits[0]) if hits else ("BACK", None)
        hits = [e for e in screen if label_hits(e.label, self.cue) and e.uid not in self.done_uids]
        if phone.clipboard is not None and phone.cur.uid not in self.done_uids \
                and label_hits(phone.cur.label, self.cue) and phone.clipboard.parent is not phone.cur:
            self.done_uids.add(phone.cur.uid)
            return ("PASTE", None)
        for e in hits:
            if e.kind == "file":
                self.done_uids.add(e.uid)
                return (self.rng.choice(self.verbs), e)
            if e.kind == "toggle":
                self.done_uids.add(e.uid)
                return ("TOGGLE", e)
        for e in hits:
            if e.kind in ("app", "folder") and e.uid not in self.opened:
                self.opened.add(e.uid)
                return ("OPEN", e)
        return ("BACK", None) if phone.cur.parent is not None else ("HOME", None)
