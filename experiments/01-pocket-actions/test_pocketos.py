"""Code-correctness tests for the PocketOS simulator, expert and baselines (< 10 s)."""
import random

import pytest

from baselines import KeywordAgent, RandomAgent
from pocketos import (SPLITS, TEMPLATES, TEST_WORDS, TRAIN_WORDS, Phone, expert_action,
                      make_task, optimal_steps, run_episode, split_seeds, task_done)


def expert(phone, task):
    return expert_action(phone, task)


@pytest.mark.parametrize("split", list(SPLITS))
def test_expert_solves_every_split(split):
    for s in split_seeds(split, 60):
        ok, steps, budget = run_episode(expert, s, split, max_factor=1.0)
        assert ok and steps == budget


def test_generation_is_deterministic():
    a, ta = make_task(123)
    b, tb = make_task(123)
    assert ta.prompt == tb.prompt
    assert [g.target for g in ta.goals] == [g.target for g in tb.goals]


def test_heldout_apps_use_unseen_names_and_templates():
    assert not set(TRAIN_WORDS) & set(TEST_WORDS)
    assert any(t.heldout for t in TEMPLATES) and any(not t.heldout for t in TEMPLATES)
    p = Phone(5, heldout=True)
    names = {a.label for a in p.root.children}
    assert names <= set(TEST_WORDS)


def test_split_seeds_disjoint():
    seen = set()
    for split in SPLITS:
        r = set(split_seeds(split, 1000))
        assert not r & seen
        seen |= r


def test_long_split_is_longer():
    short = max(optimal_steps(s, **SPLITS["id"]) for s in split_seeds("id", 100))
    long_mean = sum(optimal_steps(s, **SPLITS["long"]) for s in split_seeds("long", 100)) / 100
    assert long_mean > 2 * 10 and short < 40


def test_invalid_action_is_noop():
    phone, task = make_task(9)
    other_app = phone.root.children[0]
    phone.step("OPEN", other_app)
    stranger = phone.root.children[1]
    assert phone.step("OPEN", stranger) is False
    assert phone.cur is other_app


def test_task_is_not_done_at_start():
    for s in split_seeds("id", 50):
        phone, task = make_task(s)
        assert not task_done(phone, task)


def test_random_agent_is_near_zero():
    rng_agent = RandomAgent(0)
    wins = sum(run_episode(rng_agent, s, "id")[0] for s in split_seeds("id", 100))
    assert wins <= 10


def test_keyword_agent_runs_and_is_imperfect():
    wins = sum(run_episode(KeywordAgent(0), s, "id")[0] for s in split_seeds("id", 100))
    assert 0 < wins < 100
