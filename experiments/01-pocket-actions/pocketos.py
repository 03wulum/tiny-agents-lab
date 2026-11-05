"""PocketOS: a tiny procedural phone simulator for action-model experiments.

A phone holds apps; each app is a tree of folders holding files, plus
settings apps holding toggles. Every phone is generated from a seed, so a
policy cannot memorise layouts. Tasks are composed of sub-goals (move,
delete, share, toggle) and come with a natural-language instruction plus a
structured goal used for scoring. A scripted expert returns the optimal next
action from any state, which supplies behavioural-cloning and DAgger labels.

Pure Python, no dependencies, no side effects on import.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------- vocab ----
VERBS = ["OPEN", "BACK", "HOME", "TOGGLE", "CUT", "PASTE", "DELETE", "SHARE", "SEND"]
NO_TARGET = {"BACK", "HOME", "PASTE"}  # verbs that act on the screen, not an element
KINDS = ["app", "folder", "file", "toggle", "contact"]
FTYPES = ["pdf", "jpg", "txt", "mp3", "doc", "png", "csv", "mov"]

# Two disjoint label pools: seen-app templates draw from TRAIN_WORDS,
# held-out templates from TEST_WORDS, so names in unseen apps are never seen.
_SYL_A = ["ba", "ko", "ri", "mu", "te", "lo", "sa", "vi", "ne", "do", "pa", "zu"]
_SYL_B = ["xo", "fy", "qa", "wen", "jur", "hix", "gry", "yel", "osk", "tav", "plu", "dre"]


def _pool(syl, n, rng):
    words = set()
    while len(words) < n:
        words.add("".join(rng.choice(syl) for _ in range(rng.randint(2, 3))))
    return sorted(words)


_r = random.Random(1234)
TRAIN_WORDS = _pool(_SYL_A, 1500, _r)
TEST_WORDS = _pool(_SYL_B, 1500, _r)
CONTACT_NAMES = ["ana", "ben", "chloe", "dev", "eli", "fatima", "gus", "hana", "ivan", "jo", "kai", "lena"]
SETTING_NAMES = ["bluetooth", "wifi", "airplane", "location", "dark mode", "hotspot", "nfc", "vpn",
                 "autosync", "battery saver", "do not disturb", "auto rotate"]

# Instruction phrasing. Synonyms in *_TEST are only used by the "wording" split.
PHRASES = {
    "move": (["move {f} to {d}", "put {f} into {d}", "transfer {f} to {d}"],
             ["relocate {f} over to {d}", "shift {f} into {d}"]),
    "delete": (["delete {f}", "remove {f}", "trash {f}"],
               ["get rid of {f}", "erase {f}"]),
    "share": (["share {f} with {c}", "send {f} to {c}"],
              ["forward {f} to {c}", "pass {f} along to {c}"]),
    "toggle_on": (["turn on {s} in {a}", "enable {s} in {a}"], ["switch {s} on in {a}", "activate {s} in {a}"]),
    "toggle_off": (["turn off {s} in {a}", "disable {s} in {a}"], ["switch {s} off in {a}", "deactivate {s} in {a}"]),
}
SUPER = {  # selector words: (train words, test-only synonyms)
    "newest": (["newest", "latest"], ["most recent"]),
    "oldest": (["oldest", "earliest"], ["least recent"]),
    "largest": (["largest", "biggest"], ["heaviest"]),
    "smallest": (["smallest", "tiniest"], ["lightest"]),
}
JOINERS = (["then", "and then", "and"], ["after that", "next"])


# ---------------------------------------------------------------- world ----
@dataclass(eq=False)
class Node:
    kind: str
    label: str
    parent: "Node | None" = None
    children: list = field(default_factory=list)
    ftype: str = ""
    date: int = 0
    size: int = 0
    on: bool = False
    uid: int = 0

    def path(self):
        out, n = [], self
        while n is not None:
            out.append(n)
            n = n.parent
        return out[::-1]  # root .. self


@dataclass
class Template:
    tid: int
    kind: str          # "files" or "settings"
    depth: int         # folder nesting below the app root (files apps)
    fanout: tuple      # (min, max) folders per level
    nfiles: tuple      # (min, max) files per folder
    ftypes: tuple
    heldout: bool


def make_templates(n=20, n_heldout=6, seed=7):
    rng = random.Random(seed)
    out = []
    for t in range(n):
        kind = "settings" if t % 5 == 4 else "files"
        out.append(Template(
            tid=t, kind=kind, depth=rng.randint(1, 2),
            fanout=(rng.randint(2, 3), rng.randint(3, 4)),
            nfiles=(rng.randint(2, 3), rng.randint(4, 6)),
            ftypes=tuple(rng.sample(FTYPES, rng.randint(3, 6))),
            heldout=t >= n - n_heldout))
    return out


TEMPLATES = make_templates()


class Phone:
    """Mutable world state plus a cursor (current screen)."""

    def __init__(self, seed: int, heldout: bool = False, clutter: float = 1.0):
        rng = random.Random(seed)
        self.rng = rng
        words = list(TEST_WORDS if heldout else TRAIN_WORDS)
        rng.shuffle(words)
        self._words = iter(words)
        self._uid = 0
        self.root = self._node("home", "home")
        pool = [t for t in TEMPLATES if t.heldout == heldout]
        files_t = [t for t in pool if t.kind == "files"]
        set_t = [t for t in pool if t.kind == "settings"]
        n_apps = rng.randint(4, 7)
        chosen = [rng.choice(set_t)] + [rng.choice(files_t) for _ in range(n_apps - 1)]
        rng.shuffle(chosen)
        for t in chosen:
            app = self._node("app", self._word(), self.root)
            if t.kind == "settings":
                for s in rng.sample(SETTING_NAMES, rng.randint(4, int(min(12, 7 * clutter)))):
                    tg = self._node("toggle", s, app)
                    tg.on = rng.random() < 0.5
            else:
                self._fill(app, t, t.depth, clutter)
        self.contacts = rng.sample(CONTACT_NAMES, rng.randint(4, 7))
        self.cur = self.root
        self.clipboard: Node | None = None
        self.sharing: Node | None = None     # file whose contact picker is open
        self.shared: set = set()             # (file uid, contact)
        self.deleted: set = set()

    # -- construction
    def _node(self, kind, label, parent=None, **kw):
        self._uid += 1
        n = Node(kind, label, parent, uid=self._uid, **kw)
        if parent is not None:
            parent.children.append(n)
        return n

    def _word(self):
        return next(self._words)

    def _fill(self, parent, t, depth, clutter):
        rng = self.rng
        used = set()
        for _ in range(int(rng.randint(*t.nfiles) * clutter)):
            ft = rng.choice(t.ftypes)
            name = self._word()
            if (name, ft) in used:
                continue
            used.add((name, ft))
            self._node("file", f"{name}.{ft}", parent, ftype=ft,
                       date=rng.randint(1, 10_000), size=rng.randint(1, 10_000))
        if depth > 0:
            for _ in range(int(rng.randint(*t.fanout) * clutter)):
                f = self._node("folder", self._word(), parent)
                self._fill(f, t, depth - 1, clutter)

    # -- queries
    def find(self, uid):
        stack = [self.root]
        while stack:
            n = stack.pop()
            if n.uid == uid:
                return n
            stack.extend(n.children)
        return None

    def screen(self):
        """Elements visible on the current screen, in display order."""
        if self.sharing is not None:
            if not hasattr(self, "_contact_nodes"):
                self._contact_nodes = [Node("contact", c, uid=-(i + 1)) for i, c in enumerate(self.contacts)]
            return list(self._contact_nodes)
        return list(self.cur.children)

    def observation(self):
        return {
            "where": [n.label for n in self.cur.path()[1:]],
            "elements": self.screen(),
            "clipboard": self.clipboard,
            "sharing": self.sharing,
        }

    # -- dynamics
    def valid_actions(self):
        acts = [("BACK", None), ("HOME", None)]
        if self.clipboard is not None and self.sharing is None and self.cur.kind in ("app", "folder"):
            acts.append(("PASTE", None))
        for e in self.screen():
            if e.kind in ("app", "folder"):
                acts.append(("OPEN", e))
            elif e.kind == "file":
                acts += [("CUT", e), ("DELETE", e), ("SHARE", e)]
            elif e.kind == "toggle":
                acts.append(("TOGGLE", e))
            elif e.kind == "contact":
                acts.append(("SEND", e))
        return acts

    def step(self, verb, el=None):
        """Apply an action. Invalid actions are no-ops (they still cost a step)."""
        if verb not in VERBS or (el is None and verb not in NO_TARGET):
            return False
        if verb == "HOME":
            self.cur, self.sharing = self.root, None
        elif verb == "BACK":
            if self.sharing is not None:
                self.sharing = None
            elif self.cur.parent is not None:
                self.cur = self.cur.parent
        elif self.sharing is not None:
            if verb == "SEND" and el is not None and el.kind == "contact":
                self.shared.add((self.sharing.uid, el.label))
                self.sharing = None
        elif el is not None and el not in self.cur.children:
            return False
        elif verb == "OPEN" and el.kind in ("app", "folder"):
            self.cur = el
        elif verb == "TOGGLE" and el.kind == "toggle":
            el.on = not el.on
        elif verb == "CUT" and el.kind == "file":
            self.clipboard = el
        elif verb == "DELETE" and el.kind == "file":
            self.cur.children.remove(el)
            self.deleted.add(el.uid)
            if self.clipboard is el:
                self.clipboard = None
        elif verb == "SHARE" and el.kind == "file":
            self.sharing = el
        elif verb == "PASTE" and self.clipboard is not None and self.cur.kind in ("app", "folder"):
            f = self.clipboard
            if f.parent is not None and f in f.parent.children:
                f.parent.children.remove(f)
            f.parent = self.cur
            self.cur.children.append(f)
            self.clipboard = None
        else:
            return False
        return True


# ---------------------------------------------------------------- tasks ----
@dataclass
class SubGoal:
    op: str            # move | delete | share | toggle
    target: int        # uid of file or toggle
    dest: int = 0      # folder uid (move)
    contact: str = ""  # (share)
    value: bool = False  # (toggle)


@dataclass
class Task:
    prompt: str
    goals: list
    seed: int


def _file_ref(phone, f, rng, test_words):
    """Describe a file either by name or by superlative + type within its folder."""
    loc = " ".join(n.label for n in f.parent.path()[1:])
    sibs = [s for s in f.parent.children if s.kind == "file" and s.ftype == f.ftype]
    if rng.random() < 0.5 and len(sibs) >= 2:
        for key, attr, rev in (("newest", "date", True), ("oldest", "date", False),
                               ("largest", "size", True), ("smallest", "size", False)):
            best = sorted(sibs, key=lambda s: getattr(s, attr), reverse=rev)[0]
            if best is f and rng.random() < 0.7:
                w = rng.choice(SUPER[key][1] if test_words else SUPER[key][0])
                return f"the {w} {f.ftype} in {loc}"
    return f"{f.label} in {loc}"


def _all(phone, kind):
    out, stack = [], [phone.root]
    while stack:
        n = stack.pop()
        if n.kind == kind:
            out.append(n)
        stack.extend(n.children)
    return sorted(out, key=lambda n: n.uid)


def make_task(seed: int, n_goals=(1, 3), heldout=False, test_words=False, clutter=1.0):
    """Return (phone, task). The phone is fresh; the task is solvable."""
    phone = Phone(seed, heldout=heldout, clutter=clutter)
    rng = random.Random(seed * 7919 + 1)
    files, folders, toggles = _all(phone, "file"), _all(phone, "folder"), _all(phone, "toggle")
    goals, parts, used = [], [], set()
    k = rng.randint(*n_goals)
    while len(goals) < k:
        op = rng.choice(["move", "move", "delete", "share", "toggle"])
        if op == "toggle":
            t = rng.choice(toggles)
            if t.uid in used:
                continue
            used.add(t.uid)
            val = not t.on
            key = "toggle_on" if val else "toggle_off"
            tmpl = rng.choice(PHRASES[key][1] if test_words else PHRASES[key][0])
            parts.append(tmpl.format(s=t.label, a=t.parent.label))
            goals.append(SubGoal("toggle", t.uid, value=val))
            continue
        f = rng.choice(files)
        if f.uid in used:
            continue
        used.add(f.uid)
        fref = _file_ref(phone, f, rng, test_words)
        tmpl = rng.choice(PHRASES[op][1] if test_words else PHRASES[op][0])
        if op == "move":
            dests = [d for d in folders + _all(phone, "app")
                     if d is not f.parent and d.kind in ("folder", "app")
                     and not any(c.kind == "toggle" for c in d.children)]
            d = rng.choice(dests)
            dref = " ".join(n.label for n in d.path()[1:])
            parts.append(tmpl.format(f=fref, d=dref))
            goals.append(SubGoal("move", f.uid, dest=d.uid))
        elif op == "delete":
            parts.append(tmpl.format(f=fref))
            goals.append(SubGoal("delete", f.uid))
        else:
            c = rng.choice(phone.contacts)
            parts.append(tmpl.format(f=fref, c=c))
            goals.append(SubGoal("share", f.uid, contact=c))
    joiners = JOINERS[1] if test_words else JOINERS[0]
    prompt = parts[0]
    for p in parts[1:]:
        prompt += f" {rng.choice(joiners)} {p}"
    return phone, Task(prompt, goals, seed)


def goal_done(phone, g: SubGoal):
    if g.op == "toggle":
        return phone.find(g.target).on == g.value
    if g.op == "delete":
        return g.target in phone.deleted
    if g.op == "share":
        return (g.target, g.contact) in phone.shared
    f = phone.find(g.target)
    return f is not None and f.parent is not None and f.parent.uid == g.dest and phone.clipboard is not f


def task_done(phone, task):
    return all(goal_done(phone, g) for g in task.goals)


def task_broken(phone, task):
    """True if the task can no longer be completed (e.g. a file to move was deleted)."""
    return any(g.op in ("move", "share") and g.target in phone.deleted for g in task.goals)


# --------------------------------------------------------------- expert ----
def _nav(phone, dest: Node):
    """Next action that moves the cursor toward `dest` along the shortest path."""
    if phone.sharing is not None:
        return ("BACK", None)
    cur_path, dst_path = phone.cur.path(), dest.path()
    common = 0
    while common < min(len(cur_path), len(dst_path)) and cur_path[common] is dst_path[common]:
        common += 1
    ups = len(cur_path) - common
    if ups > 0:
        # Via HOME: 1 + (len(dst)-1) steps. Via BACKs: ups + (len(dst)-common) steps.
        return ("HOME", None) if common < ups else ("BACK", None)
    return ("OPEN", dst_path[len(cur_path)])


def expert_action(phone, task):
    """Optimal next action for the first unfinished sub-goal (in prompt order)."""
    for g in task.goals:
        if goal_done(phone, g):
            continue
        if g.op == "toggle":
            t = phone.find(g.target)
            return ("TOGGLE", t) if phone.cur is t.parent and phone.sharing is None else _nav(phone, t.parent)
        if g.op == "share":
            f = phone.find(g.target)
            if phone.sharing is f:
                c = next(e for e in phone.screen() if e.label == g.contact)
                return ("SEND", c)
            return ("SHARE", f) if phone.cur is f.parent and phone.sharing is None else _nav(phone, f.parent)
        if g.op == "delete":
            f = phone.find(g.target)
            return ("DELETE", f) if phone.cur is f.parent and phone.sharing is None else _nav(phone, f.parent)
        if g.op == "move":
            f = phone.find(g.target)
            dest = phone.find(g.dest)
            if phone.clipboard is f:
                return ("PASTE", None) if phone.cur is dest and phone.sharing is None else _nav(phone, dest)
            return ("CUT", f) if phone.cur is f.parent and phone.sharing is None else _nav(phone, f.parent)
    return None


def optimal_steps(seed, **kw):
    phone, task = make_task(seed, **kw)
    n = 0
    while not task_done(phone, task):
        v, e = expert_action(phone, task)
        phone.step(v, e)
        n += 1
        assert n < 200, "expert failed to finish"
    return n


SPLITS = {
    # name: make_task kwargs. Seeds are disjoint per split (see split_seeds).
    "train": dict(n_goals=(1, 3)),
    "id": dict(n_goals=(1, 3)),
    "unseen_apps": dict(n_goals=(1, 3), heldout=True),
    "long": dict(n_goals=(4, 6)),
    "wording": dict(n_goals=(1, 3), test_words=True),
}
_SPLIT_BASE = {"train": 0, "id": 10_000_000, "unseen_apps": 20_000_000, "long": 30_000_000, "wording": 40_000_000}


def split_seeds(split, n, offset=0):
    b = _SPLIT_BASE[split] + offset
    return range(b, b + n)


def run_episode(policy, seed, split="id", max_factor=2.0, **overrides):
    """Roll out `policy(phone, task) -> (verb, element)`. Success = goal reached
    within max_factor x the optimal step count."""
    kw = {**SPLITS[split], **overrides}
    budget = int(max_factor * optimal_steps(seed, **kw))
    phone, task = make_task(seed, **kw)
    for t in range(budget):
        v, e = policy(phone, task)
        phone.step(v, e)
        if task_done(phone, task):
            return True, t + 1, budget
        if task_broken(phone, task):
            break
    return False, budget, budget
