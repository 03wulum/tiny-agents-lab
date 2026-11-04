# 01 · Pocket actions: a tiny model that operates apps without language

**Goal:** finish multi-step tasks in phone apps the model has never seen, with 2M parameters or less (8 MB), trained in about 5 minutes (10 at most) on one GPU.
**Verdict:** PENDING. The code and tests are done, and training on the GPU is next.
**Details:** [plan](https://claude.ai/artifact/41b5MpqiRaq4BaRwsBHPch) · [results](results/results.md) · [explainer](EXPLAINER.md)

| # | Criterion | Threshold | Result |
| --- | --- | --- | --- |
| C1 | Budget: kept models, GPU training, eval RAM, CPU time per decision | ≤ 50 MB, ≤ 10 min, ≤ 2 GB, ≤ 10 ms | pending |
| C2 | In-distribution success (2M model, 3 seeds) | ≥ 90% | pending |
| C3 | Success on unseen apps (core) | ≥ 70% and ≥ keyword + 25 pts | pending |
| C4 | Success on tasks with twice the sub-goals | ≥ 60% | pending |
| C5 | Cue model vs same-size text model, unseen apps | ≥ +15 pts | pending |
| NC | Negative control: the instruction swapped for another task's | must stay < 15% | pending |

## The idea

PocketOS is a simulated phone whose apps, folders and file names are generated fresh for every task, so nothing can be memorised. The model reads the instruction as loose cue words (each tagged with its clause and its role, for example "location" or "destination") and sees the screen as a list of elements. Every step, it points at one element and one action. It never writes text. It is compared against a same-size model that reads and writes plain text, a keyword-matching baseline (27% on unseen apps) and a random agent (1%).

Prior work: [FunctionGemma](https://blog.google/technology/developers/functiongemma/) (270M, 85% on Mobile Actions), [Octopus v2](https://arxiv.org/abs/2404.01744), [TinyAgent](https://arxiv.org/abs/2409.00608), [Tiny Recursive Model](https://arxiv.org/abs/2510.04871), [BabyAI](https://arxiv.org/abs/1810.08272).

## Run it

```bash
python train.py --smoke   # 1-2 min end-to-end check (NVIDIA GPU)
python train.py           # full sweep: about 5 min of GPU training (hard cap 5.5), then GPU evaluation
python run.py             # PASS/FAIL table, plots, results/ (CPU, never trains)
python -m pytest -q
```

Training needs `requirements-gpu.txt` (PyTorch with CUDA). Evaluation, tests and CI use only NumPy.

## Deviations from SOP

- Training needs a GPU, so `run.py` judges the GPU evaluation in `results/eval.json` and re-checks a sample of it on CPU. It doesn't recompute every number.
- Obi cut the training budget to about 5 minutes (2026-11-04), so the recursive-2M, no-match-bits, demonstration-count and 500K text-model runs were dropped before any training. The blind-prompt, random-weights, int8 and clutter controls remain, and they are evaluation-only.
- The cue parser knows only the training-time joiner words ("then", "and"). The "new wording" split uses joiners it has never seen.

## Files

`pocketos.py` simulator and expert · `features.py` cue and text encodings · `model_np.py` NumPy inference · `model_torch.py` training models · `rollout.py` demonstrations and DAgger · `train.py` GPU training and evaluation · `criteria.py` pre-registered criteria · `run.py` verdict
