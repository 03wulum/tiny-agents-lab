"""Code-correctness tests for features and the models (< 10 s). Torch parity runs only where torch is installed."""
import numpy as np
import pytest

import model_np
from features import (CueState, TextState, action_index, collate, decode_action, encode_cue, encode_text,
                      padded_index, parse_cues, parse_text_action, text_target, unpad_index)
from pocketos import expert_action, make_task, run_episode, split_seeds, task_done

CFG = dict(kind="cue", d=16, layers=2, heads=2, loops=1, no_match=False)


def test_parse_cues_roles_and_clauses():
    words, roles, clauses = parse_cues("move a.pdf in x y to z then delete b.txt in w")
    assert words[1] == "a.pdf"
    assert roles[words.index("x")] == 1 and roles[words.index("z")] == 2
    assert clauses[words.index("b.txt")] == 1 and clauses[words.index("z")] == 0


def test_expert_actions_roundtrip_through_encoding():
    for s in split_seeds("id", 20):
        phone, task = make_task(s)
        cs = CueState(task.prompt)
        while not task_done(phone, task):
            verb, el = expert_action(phone, task)
            enc, els = encode_cue(phone, cs)
            flat = action_index(enc, els, verb, el)
            assert enc["valid"].reshape(-1)[flat]
            assert decode_action(enc, els, flat) == (verb, el)
            Lc = len(enc["cue_id"]) + 3
            assert unpad_index(enc, padded_index(enc, flat, Lc), Lc) == flat
            cs.update(phone.cur, verb, el)
            phone.step(verb, el)


def test_ticks_mark_finished_clauses():
    phone, task = make_task(3)
    cs = CueState(task.prompt)
    while not task_done(phone, task):
        verb, el = expert_action(phone, task)
        cs.update(phone.cur, verb, el)
        phone.step(verb, el)
    assert cs.ticks.sum() >= 1


def test_text_target_parses_back():
    phone, task = make_task(5)
    verb, el = expert_action(phone, task)
    ids, _ = encode_text(phone, TextState(task.prompt))
    assert ids[-1] == 128 and len(ids) <= 1024
    assert parse_text_action(phone, text_target(verb, el)[:-1])[0] == (verb, el)
    assert parse_text_action(phone, [ord(c) for c in "OPEN nosuchthing"])[1] is False


def test_collate_masks_padding():
    encs = []
    for s in (1, 2):
        phone, task = make_task(s)
        encs.append(encode_cue(phone, CueState(task.prompt))[0])
    b = collate(encs)
    assert b["pad"].shape == b["valid"].shape[:2]
    assert not (b["valid"] & b["pad"][..., None]).any()


def test_numpy_policy_runs_and_respects_validity():
    P = model_np.random_params(CFG)
    agent = model_np.CueAgent(CFG, P)
    for s in split_seeds("id", 5):
        run_episode(agent, s, "id")
    assert model_np.n_params(P) < 40_000


def test_int8_is_close_and_smaller():
    P = model_np.random_params(CFG)
    Q, nbytes = model_np.quantize_int8(P)
    assert nbytes < 4 * model_np.n_params(P) / 2
    assert max(np.abs(P[k] - Q[k]).max() for k in P) < 0.01


def test_numpy_matches_torch():
    torch = pytest.importorskip("torch")
    import model_torch as MT
    m = MT.CuePolicy(16, 2, 2).eval()
    P = {k: v.detach().numpy() for k, v in m.state_dict().items()}
    assert set(P) == set(model_np.random_params(CFG))
    encs = []
    for s in (1, 2, 3):
        phone, task = make_task(s)
        encs.append(encode_cue(phone, CueState(task.prompt))[0])
    b = collate(encs)
    with torch.no_grad():
        t = m(MT.to_torch(b, "cpu")).numpy()
    n = model_np.cue_forward(m.cfg, P, b)
    fin = np.isfinite(n)
    assert (fin == np.isfinite(t)).all()
    assert np.allclose(n[fin], t[fin], atol=1e-4)


def test_lockstep_rollout_matches_single_episodes():
    from baselines import KeywordAgent
    from rollout import Single, rollout
    seeds = list(split_seeds("id", 30))
    batched = rollout(Single(lambda: KeywordAgent(0)), seeds, "id")
    single = [run_episode(KeywordAgent(0), s, "id")[0] for s in seeds]
    assert batched == single


def test_dagger_collection_labels_learner_states():
    from baselines import RandomAgent
    from rollout import Single, rollout
    succ, lc, lt = rollout(Single(lambda: RandomAgent(0)), list(split_seeds("train", 5)), "train", collect=True)
    assert len(lc) == len(lt) > 0
    for enc, flat in lc:
        assert enc["valid"].reshape(-1)[flat]


def test_make_demos_small():
    from rollout import make_demos
    cue, text = make_demos(4, procs=2)
    assert len(cue) == len(text) > 4
