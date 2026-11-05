"""Turn a PocketOS state + instruction into model inputs.

Two encodings, one per model family:

* cue encoding (cue-and-pointer policy): the instruction becomes a set of cue
  words, each tagged with its clause number, a role (other / location /
  destination) and a "ticked" bit the agent sets once it has acted on that
  word. Screen elements become feature vectors plus a binary match matrix
  (which cue words equal which element labels). No word order is modelled and
  no text is generated.
* text encoding (text-token baseline): the same information serialized as one
  ASCII string; the model must write its action as text.

Pure NumPy / stdlib, shared by training (GPU) and evaluation (CPU).
"""
from __future__ import annotations

import re
import zlib

import numpy as np

from pocketos import FTYPES, KINDS, VERBS

H = 1024              # hash buckets for cue words (id 0 = unknown)
MAX_CLAUSE = 8
ROLE = {"other": 0, "loc": 1, "dst": 2}
JOIN_WORDS = {"then", "and"}            # clause separators known at train time
ROLE_WORDS = {"in": "loc", "to": "dst", "into": "dst", "with": "dst"}
N_EL_FEATS = 10
N_ST_FEATS = 6
VERB_ID = {v: i for i, v in enumerate(VERBS)}
SCREEN_VERBS = ("BACK", "HOME", "PASTE")
EFFECT_VERBS = {"CUT", "DELETE", "SHARE", "SEND", "TOGGLE", "PASTE"}
TERMINAL_VERBS = {"PASTE", "DELETE", "SEND", "TOGGLE"}  # the last action of each kind of sub-goal


def word_hash(w: str) -> int:
    return 1 + zlib.crc32(w.encode()) % (H - 1)


def parse_cues(prompt: str, fixed=False):
    """Return (words, role ids, clause ids) for an instruction.

    fixed=False is experiment 01's parser: every joiner word starts a new clause ("and then"
    counts twice) and ids are clipped at MAX_CLAUSE - 1. fixed=True (experiment 02) starts a new
    clause only after a non-joiner word and does not clip (ids are made relative later)."""
    words = re.findall(r"[a-z0-9]+(?:\.[a-z0-9]+)?", prompt.lower())
    roles, clauses = [], []
    clause, role, content = 0, ROLE["other"], False
    for w in words:
        if w in JOIN_WORDS:
            if fixed:
                if content:
                    clause, role, content = clause + 1, ROLE["other"], False
            else:
                clause, role = min(clause + 1, MAX_CLAUSE - 1), ROLE["other"]
        else:
            content = True
            if w in ROLE_WORDS:
                role = ROLE[ROLE_WORDS[w]]
        roles.append(role)
        clauses.append(clause)
    return words, np.array(roles, np.int64), np.array(clauses, np.int64)


def label_words(node) -> list:
    return node.label.split() if node.kind == "toggle" else [node.label]


class CueState:
    """Per-episode agent memory: parsed cues, ticks and the previous verb."""

    def __init__(self, prompt: str, relative: bool = False):
        self.relative = relative
        self.words, self.roles, self.clauses = parse_cues(prompt, fixed=relative)
        self.ids = np.array([word_hash(w) for w in self.words], np.int64)
        self.ticks = np.zeros(len(self.words), np.int64)
        self.done = set()       # clauses finished by a terminal action (relative mode)
        self.prev = len(VERBS)  # "none"

    def _tick(self, words, role=None):
        words = set(words)
        hit = [i for i, w in enumerate(self.words)
               if w in words and not self.ticks[i] and (role is None or self.roles[i] == role)
               and self.clauses[i] not in self.done]  # done stays empty in 01's absolute mode
        if not hit:
            return None
        c = min(self.clauses[i] for i in hit)
        for i in hit:
            if self.clauses[i] == c:
                self.ticks[i] = 1
        return int(c)

    def update(self, phone_before_cur, verb, el):
        """Call after choosing an action (before or after stepping; uses the pre-step screen)."""
        self.prev = VERB_ID[verb]
        c = None
        if verb in ("CUT", "DELETE", "SHARE", "TOGGLE", "SEND") and el is not None:
            c = self._tick(label_words(el), ROLE["dst"] if verb == "SEND" else None)
        elif verb == "PASTE":
            c = self._tick([phone_before_cur.label], ROLE["dst"])
        if verb in TERMINAL_VERBS and self.relative:
            if c is None:  # target named by description ("the latest mov"): finish the current clause
                c = self.current()
            self.done.add(c)

    def current(self):
        """Lowest clause not yet finished (one past the last clause when all are finished)."""
        n = int(self.clauses.max()) + 1 if len(self.clauses) else 0
        return next((c for c in range(n) if c not in self.done), n)

    def clause_feature(self):
        """Clause ids fed to the model. Absolute (01) or relative to the current clause (02):
        0 = finished, 1 = current (lowest unfinished), 2 = next, 3 = later."""
        if not self.relative:
            return self.clauses.copy()
        rel = np.clip(self.clauses - self.current(), -1, 2) + 1
        rel[np.isin(self.clauses, list(self.done))] = 0
        return rel.astype(np.int64)


def _rank_flags(elements):
    """newest / oldest / largest / smallest among same-type files on screen."""
    flags = {}
    by_type = {}
    for e in elements:
        if e.kind == "file":
            by_type.setdefault(e.ftype, []).append(e)
    for group in by_type.values():
        if len(group) < 2:
            continue
        for k, (attr, rev) in enumerate((("date", True), ("date", False), ("size", True), ("size", False))):
            best = sorted(group, key=lambda s: getattr(s, attr), reverse=rev)[0]
            flags.setdefault(id(best), set()).add(k)
    return flags, by_type


def encode_cue(phone, cs: CueState):
    """Arrays for one decision. Token order: [state, cues..., elements...]."""
    els = phone.screen()
    nc, ne = len(cs.words), len(els)
    wset = [set([w]) for w in cs.words]
    flags, by_type = _rank_flags(els)
    kind = np.array([KINDS.index(e.kind) for e in els], np.int64)
    ftype = np.array([FTYPES.index(e.ftype) + 1 if e.ftype else 0 for e in els], np.int64)
    ef = np.zeros((ne, N_EL_FEATS), np.float32)
    match = np.zeros((ne, nc), np.float32)
    for j, e in enumerate(els):
        lw = set(label_words(e))
        for k in flags.get(id(e), ()):
            ef[j, k] = 1
        ef[j, 4] = float(e.kind == "toggle" and e.on)
        ef[j, 5] = float(phone.clipboard is e)
        ef[j, 6] = float(e.kind == "file" and len(by_type.get(e.ftype, [])) >= 2)
        for i in range(nc):
            if cs.words[i] in lw:
                match[j, i] = 1
        ef[j, 7] = float(match[j].any())
        ef[j, 8] = float((match[j] * cs.ticks).any())
        ef[j, 9] = float(e.kind == "contact")
    depth = len(phone.cur.path()) - 1
    sf = np.zeros(N_ST_FEATS, np.float32)
    sf[0] = float(phone.clipboard is not None)
    sf[1] = float(phone.sharing is not None)
    sf[2 + min(depth, 3)] = 1.0
    path_words = set()
    for n in phone.cur.path()[1:]:
        path_words |= set(label_words(n))
    pmatch = np.array([float(w in path_words) for w in cs.words], np.float32)
    clip_words = set(label_words(phone.clipboard)) if phone.clipboard is not None else set()
    cmatch = np.array([float(w in clip_words) for w in cs.words], np.float32)
    # valid-action mask over 9 verbs for every token row
    valid = np.zeros((1 + nc + ne, len(VERBS)), bool)
    index = {id(e): j for j, e in enumerate(els)}
    for v, e in phone.valid_actions():
        if e is None:
            valid[0, VERB_ID[v]] = True
        else:
            valid[1 + nc + index[id(e)], VERB_ID[v]] = True
    del wset
    return dict(cue_id=cs.ids.copy(), cue_role=cs.roles.copy(), cue_clause=cs.clause_feature(),
                cue_tick=cs.ticks.copy(), el_kind=kind, el_ftype=ftype, el_feat=ef,
                match=match, st_feat=sf, path_match=pmatch, clip_match=cmatch,
                prev=np.int64(cs.prev), valid=valid), els


def action_index(enc, els, verb, el):
    """Flat index (row * n_verbs + verb) of an action in the valid mask."""
    nc = len(enc["cue_id"])
    row = 0 if el is None else 1 + nc + next(j for j, e in enumerate(els) if e is el)
    return row * len(VERBS) + VERB_ID[verb]


def decode_action(enc, els, flat):
    nv = len(VERBS)
    row, v = divmod(int(flat), nv)
    nc = len(enc["cue_id"])
    return (VERBS[v], None) if row == 0 else (VERBS[v], els[row - 1 - nc])


def collate(batch, Lc=None, Le=None):
    """Pad a list of encodings into batch arrays. Rows: [state, cues(Lc), elements(Le)]."""
    B = len(batch)
    Lc = Lc or max(1, max(len(b["cue_id"]) for b in batch))
    Le = Le or max(1, max(len(b["el_kind"]) for b in batch))
    T = 1 + Lc + Le
    out = dict(
        cue_id=np.zeros((B, Lc), np.int64), cue_role=np.zeros((B, Lc), np.int64),
        cue_clause=np.zeros((B, Lc), np.int64), cue_tick=np.zeros((B, Lc), np.int64),
        el_kind=np.zeros((B, Le), np.int64), el_ftype=np.zeros((B, Le), np.int64),
        el_feat=np.zeros((B, Le, N_EL_FEATS), np.float32), match=np.zeros((B, Le, Lc), np.float32),
        st_feat=np.zeros((B, N_ST_FEATS), np.float32), path_match=np.zeros((B, Lc), np.float32),
        clip_match=np.zeros((B, Lc), np.float32), prev=np.zeros(B, np.int64),
        pad=np.ones((B, T), bool), valid=np.zeros((B, T, len(VERBS)), bool))
    for i, b in enumerate(batch):
        nc, ne = len(b["cue_id"]), len(b["el_kind"])
        for k in ("cue_id", "cue_role", "cue_clause", "cue_tick", "path_match", "clip_match"):
            out[k][i, :nc] = b[k]
        for k in ("el_kind", "el_ftype", "el_feat"):
            out[k][i, :ne] = b[k]
        out["match"][i, :ne, :nc] = b["match"]
        out["st_feat"][i] = b["st_feat"]
        out["prev"][i] = b["prev"]
        out["pad"][i, 0] = False
        out["pad"][i, 1:1 + nc] = False
        out["pad"][i, 1 + Lc:1 + Lc + ne] = False
        v = b["valid"]
        out["valid"][i, 0] = v[0]
        out["valid"][i, 1:1 + nc] = v[1:1 + nc]
        out["valid"][i, 1 + Lc:1 + Lc + ne] = v[1 + nc:]
    return out


def padded_index(enc, flat, Lc):
    """Map a flat action index from an unpadded encoding to the padded layout."""
    nv = len(VERBS)
    row, v = divmod(int(flat), nv)
    nc = len(enc["cue_id"])
    if row > nc:
        row = row - nc + Lc
    return row * nv + v


def unpad_index(enc, flat, Lc):
    nv = len(VERBS)
    row, v = divmod(int(flat), nv)
    nc = len(enc["cue_id"])
    if row > Lc:
        row = row - Lc + nc
    return row * nv + v


# ------------------------------------------------------------ text model ----
TEXT_VOCAB = 130          # ASCII 0..127, 128 = separator, 129 = end
SEP, EOS = 128, 129
TEXT_CTX = 1024
TEXT_OUT = 40


class TextState:
    def __init__(self, prompt: str):
        self.prompt = prompt.lower()
        self.history = []

    def update(self, phone_before_cur, verb, el):
        self.history.append(verb if el is None else f"{verb} {el.label}")
        self.history = self.history[-8:]


def encode_text(phone, ts: TextState):
    els = phone.screen()
    flags, _ = _rank_flags(els)
    names = ["newest", "oldest", "largest", "smallest"]
    items = []
    for e in els:
        tags = [names[k] for k in sorted(flags.get(id(e), ()))]
        if e.kind == "toggle":
            tags.append("on" if e.on else "off")
        items.append(f"{e.kind} {e.label}" + (f" [{' '.join(tags)}]" if tags else ""))
    where = "/".join(n.label for n in phone.cur.path()[1:]) or "home"
    s = (f"task: {ts.prompt} | at: {where} | clip: {phone.clipboard.label if phone.clipboard else '-'}"
         f" | sharing: {phone.sharing.label if phone.sharing else '-'}"
         f" | last: {'; '.join(ts.history) or '-'} | screen: {'; '.join(items)}")
    ids = [min(ord(c), 127) for c in s][-(TEXT_CTX - TEXT_OUT - 1):]
    return ids + [SEP], els


def text_target(verb, el):
    s = verb if el is None else f"{verb} {el.label}"
    return [min(ord(c), 127) for c in s] + [EOS]


def parse_text_action(phone, out_ids):
    s = "".join(chr(i) for i in out_ids if i < 128).strip()
    verb, _, label = s.partition(" ")
    if verb not in VERB_ID:
        return ("NOOP", None), False
    if verb in SCREEN_VERBS and verb != "PASTE" or (verb == "PASTE" and not label):
        return (verb, None), True
    for e in phone.screen():
        if e.label == label:
            return (verb, e), True
    return ("NOOP", None), False
