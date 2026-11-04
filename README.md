# tiny-agents-lab

Tiny on-device agents: models of 50 MB or less that act inside apps (pick, move, share, toggle) without chatting or writing text.

## Results

| # | Goal | Verdict | Criteria | Finding in one line | |
| --- | --- | --- | --- | --- | --- |
| 01 | Finish multi-step tasks in phone apps never seen before, with a model of 2M parameters or less | Pending (training next) | 0/5 | Keyword baseline: 28% on unseen apps, random: 1% | [details](experiments/01-pocket-actions) |

## How this repo works

- Every experiment has pass/fail criteria fixed before any training ([SOP](docs/SOP.md)).
- Training runs on one NVIDIA GPU in 15 minutes or less. Evaluation and CI run on any CPU.
- `python run_all.py` evaluates everything and prints one verdict table.

## License

MIT
