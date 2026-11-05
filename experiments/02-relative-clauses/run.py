"""Evaluate experiment 02 and print the PASS/FAIL table. Never trains (training: train.py on GPU).

    python run.py           # judge results/eval.json, re-check a sample on CPU, write results/
    python run.py --quick   # smaller CPU re-check (< 30 s), writes nothing; what CI runs
    python run.py --clean   # delete caches and all checkpoints (keeps results/)
"""
from _shared import HERE, load01

R = load01("run")
R.HERE, R.CKPT, R.RESULTS = HERE, HERE / "checkpoints", HERE / "results"
R.TITLE = "experiment 02 (relative clauses)"

if __name__ == "__main__":
    R.main()
