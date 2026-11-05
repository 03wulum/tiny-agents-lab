"""Run every experiment's run.py and print one verdict table.

    python run_all.py           # quick CPU re-check of every experiment; never overwrites committed results
    python run_all.py --full    # full evaluation; rewrites each experiment's results/
    python run_all.py --clean   # delete caches and checkpoints (keeps results/)
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="full evaluation; rewrites results/")
    mode.add_argument("--clean", action="store_true", help="delete caches and checkpoints")
    args = ap.parse_args()
    rows = []
    exps = ROOT / "experiments"
    for exp in sorted(exps.iterdir()) if exps.exists() else []:
        if not (exp / "run.py").exists():
            continue
        flag = ["--clean"] if args.clean else ([] if args.full else ["--quick"])
        r = subprocess.run([sys.executable, "run.py"] + flag, cwd=exp)
        if r.returncode != 0:
            sys.exit(f"{exp.name}: run.py failed")
        v = exp / "results" / "verdict.json"
        verdict = json.loads(v.read_text()) if v.exists() and not args.clean else {}
        rows.append((exp.name, verdict.get("verdict", "PENDING" if not args.clean else "-"),
                     verdict.get("criteria_passed", "-")))
    print("\n| Experiment | Verdict | Criteria |\n| --- | --- | --- |")
    for name, verdict, passed in rows:
        print(f"| {name} | {verdict} | {passed} |")


if __name__ == "__main__":
    main()
