"""Tests for experiment 02's changes (relative clauses, parser fix, float16 checkpoints)."""
import numpy as np
import pytest

import model_np
from features import CueState, encode_cue, parse_cues
from pocketos import SPLITS, expert_action, make_task, split_seeds, task_done, run_episode


def test_fixed_parser_counts_and_then_once():
    words, _, old = parse_cues("delete a.txt in x and then delete b.txt in y then delete c.txt in z")
    _, _, new = parse_cues("delete a.txt in x and then delete b.txt in y then delete c.txt in z", fixed=True)
    assert old[words.index("c.txt")] == 3          # 01 behaviour kept by default
    assert new[words.index("b.txt")] == 1 and new[words.index("c.txt")] == 2


def test_fixed_parser_does_not_clip_long_prompts():
    prompt = " then ".join(f"delete f{i}.txt in x" for i in range(10))
    words, _, c = parse_cues(prompt, fixed=True)
    assert c[words.index("f9.txt")] == 9


def test_relative_clause_ids_follow_progress():
    for seed in split_seeds("long", 10):
        phone, task = make_task(seed, **SPLITS["long"])
        cs = CueState(task.prompt, relative=True)
        assert set(cs.clause_feature()) <= {1, 2, 3} and cs.clause_feature()[0] == 1
        finished = 0
        while not task_done(phone, task):
            v, e = expert_action(phone, task)
            cs.update(phone.cur, v, e)
            phone.step(v, e)
            rel = cs.clause_feature()
            assert set(rel) <= {0, 1, 2, 3}
            if v in ("PASTE", "DELETE", "SEND", "TOGGLE"):
                finished += 1
                assert len(cs.done) == finished
        assert (cs.clause_feature() == 0).all()      # every clause finished
        assert finished == len(task.goals)


def test_absolute_mode_is_unchanged_for_experiment_01():
    phone, task = make_task(split_seeds("id", 1)[0], **SPLITS["id"])
    cs = CueState(task.prompt)
    enc, _ = encode_cue(phone, cs)
    assert (enc["cue_clause"] == parse_cues(task.prompt)[2]).all()


def test_float16_checkpoint_loads_as_float32(tmp_path):
    cfg = dict(kind="cue", d=16, layers=2, heads=2, loops=1, no_match=False, relative=True)
    P = model_np.random_params(cfg)
    path = tmp_path / "m.npz"
    import json
    np.savez(path, config=np.array(json.dumps(cfg)), **{k: v.astype(np.float16) for k, v in P.items()})
    cfg2, P2 = model_np.load(path)
    assert cfg2["relative"] and all(v.dtype == np.float32 for v in P2.values())
    assert max(np.abs(P2[k] - P[k]).max() for k in P) < 1e-3
    ok = [run_episode(model_np.CueAgent(cfg2, P2), s, "long")[0] for s in split_seeds("long", 3)]
    assert len(ok) == 3


def test_torch_relative_model_saves_float16(tmp_path):
    pytest.importorskip("torch")
    import model_torch as MT
    m = MT.CuePolicy(16, 2, 2, relative=True)
    MT.save_npz(m, tmp_path / "a.npz", "float16")
    MT.save_npz(m, tmp_path / "b.npz")
    assert (tmp_path / "a.npz").stat().st_size < 0.6 * (tmp_path / "b.npz").stat().st_size
    cfg, _ = model_np.load(tmp_path / "a.npz")
    assert cfg["relative"] is True
