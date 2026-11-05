# tiny-agents-lab

Tiny on-device agents: models of 50 MB or less that act inside apps (pick, move, share, toggle) without chatting or writing text.

## Results

| # | Goal | Verdict | Criteria | Finding in one line | |
| --- | --- | --- | --- | --- | --- |
| 01 | Finish multi-step tasks in phone apps never seen before, with a model of 2M parameters or less | INCONCLUSIVE ✨ INTRIGUING | 2/5 | A 25K-parameter (100 KB) model completes 82% of tasks in unseen apps (keyword 31%); long tasks fail (5%) | [details](experiments/01-pocket-actions) |
| 02 | Fix 01's long-task failure by telling the model which instruction clause it is on | PASS ✨ INTRIGUING | 5/5 | Long tasks 5% → 65%; a 25K-parameter (62 KB) model completes 95% of tasks in unseen apps | [details](experiments/02-relative-clauses) |

## How this repo works

- Every experiment has pass/fail criteria fixed before any training ([SOP](docs/SOP.md)).
- Training runs on one NVIDIA GPU, one experiment at a time, in about 5 minutes (never more than 10). Evaluation and CI run on any CPU.
- `python run_all.py` evaluates everything and prints one verdict table.

## License

MIT
