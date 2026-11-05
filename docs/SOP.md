# Experiment SOP (summary of v0.6)

The canonical SOP is the [Experiment SOP doc](https://claude.ai/artifact/BfqrH1AaWwEYeSweLHmsCF) (v0.6, 2026-11-04). This is a short copy of the rules this repo follows.

- **Plan first.** Before coding: a hypothesis with a number, 2 to 5 pass criteria with thresholds taken from a paper or baseline (never from our own results), a baseline, a negative control that should fail, and perturbations. Obi approves the plan.
- **Verdicts.** PASS (all criteria pass and the negative control fails), FAIL (the core criterion fails), INCONCLUSIVE (mixed, or the negative control passed). An optional ✨ INTRIGUING tag with a one-line note never changes the verdict.
- **Two layers.** `pytest` checks code correctness and must stay green. The verdict checks the claim, and FAIL never breaks CI.
- **Budgets.** Everything kept (trained models, checkpoints) totals 50 MB or less per experiment. Temporary downloads are 2 GB or less and deleted by `--clean`. Training takes about 5 minutes (10 at most; Obi, 2026-11-04), one experiment at a time on the GPU, and runs only on Fortran's NVIDIA GPU, using as much free GPU capacity as is available. CI and `run.py --quick` only load checkpoints and evaluate.
- **Safety.** Install only into the repo's `.venv`. Use pinned, open-licensed packages and data. Never run downloaded code. Downloads go in a git-ignored `data/` folder.
- **READMEs.** The repo README opens with a Results table (goal, verdict, criteria passed, one-line finding, details link). Each experiment README opens with the goal, the verdict and a Details link, and stays about one phone screen long. Detail lives in `EXPLAINER.md`, `results/results.md` and `docs/`.
- **Layout.** `experiments/NN-slug/` holds README, EXPLAINER, `criteria.py`, `run.py` (`--quick`, `--seed`, `--clean`), the method modules, tests and `results/`. `run_all.py` runs everything.
- **End of each experiment.** A post-readiness verdict and a draft post of 280 characters or less. Obi decides whether to post. Run `--clean` on Fortran and report the disk space it freed.
