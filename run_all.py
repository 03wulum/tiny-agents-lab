"""Run every experiment's run.py (--quick by default) and print one verdict table.

    python run_all.py           # quick evaluation of every experiment
    python run_all.py --full    # full evaluation
    python run_all.py --clean   # delete downloads, caches and intermediate checkpoints
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def main():
    full = "--full" in sys.argv
    clean = "--clean" in sys.argv
    rows = []
    exps = ROOT / "experiments"
    for exp in sorted(exps.iterdir()) if exps.exists() else []:
        if not (exp / "run.py").exists():
            continue
        args = [sys.executable, "run.py"] + (["--clean"] if clean else ([] if full else ["--quick"]))
        r = subprocess.run(args, cwd=exp)
        if r.returncode != 0:
            sys.exit(f"{exp.name}: run.py failed")
        v = exp / "results" / "verdict.json"
        verdict = json.loads(v.read_text()) if v.exists() and not clean else {}
        rows.append((exp.name, verdict.get("verdict", "-"), verdict.get("criteria_passed", "-")))
    print("\n| Experiment | Verdict | Criteria |\n| --- | --- | --- |")
    for name, verdict, passed in rows:
        print(f"| {name} | {verdict} | {passed} |")


if __name__ == "__main__":
    main()
