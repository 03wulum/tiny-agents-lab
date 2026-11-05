# 02 · Relative clauses: fixing long tasks for the pocket action model

**Goal:** fix [experiment 01](../01-pocket-actions)'s long-task failure (5%) by telling the model which part of the instruction it is on, store models in float16, and keep generalising to unseen apps.
**Verdict:** PENDING (criteria pre-registered 2026-11-05; GPU run queued on Fortran).
**Details:** [criteria](criteria.py) · [explainer](EXPLAINER.md) · results (after the run)

| # | Criterion | Threshold |
| --- | --- | --- |
| C1 | Budget: kept models (float16), GPU training, eval RAM, CPU time per decision | ≤ 50 MB, ≤ 10 min, ≤ 2 GB, ≤ 10 ms |
| C2 | In-distribution success (2M model, 3 seeds) | ≥ 90% |
| C3 | Success on unseen apps (core) | ≥ 70% and ≥ keyword + 25 pts |
| C4 | Success on tasks with twice the sub-goals | ≥ 60% |
| C5 | Cue model vs same-size text model, unseen apps | ≥ +15 pts |
| NC | Negative control: the instruction swapped for another task's | must stay < 15% |

Same thresholds as 01. Ablation (reported, not a criterion): the 2M model with every cue shown as "current clause". If the new signal is what fixes long tasks, its long-task score should fall back toward 01's.

## What changed from 01

1. **Relative clause ids.** In 01, every cue word carried its clause number (0, 1, 2, ...). Training used at most 3 sub-goals, so the clause numbers in 4-6 sub-goal instructions were never seen. Now each word is tagged *finished*, *current*, *next* or *later*. The current clause is the first one not yet finished, and a clause finishes when the agent does its last action (paste, delete, send or toggle). Every long task looks like a short one from the model's point of view.
2. **Parser fix.** "and then" counted as two clause breaks in 01. It now counts as one.
3. **float16 checkpoints.** Half the size. 01 showed even int8 loses nothing.
4. **Longer training for the 2M models** (30 s each instead of 20 s, for C2), and one text-model seed instead of three (all three scored 0% in 01). Total planned training is about 3 minutes, plus DAgger rollouts.

The code is 01's, with the new behaviour behind switches that are off by default, so 01 still reproduces exactly. This folder only holds the settings, the criteria and the tests.

## Run it

```bash
python train.py --smoke   # 1-2 min end-to-end check (NVIDIA GPU)
python train.py           # full sweep, about 4 min of GPU training (hard cap 5.5), then GPU evaluation
python run.py             # PASS/FAIL table, plots, results/ (CPU, never trains)
python -m pytest -q
```
