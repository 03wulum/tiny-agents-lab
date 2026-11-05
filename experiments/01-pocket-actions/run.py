"""Evaluate experiment 01 and print the PASS/FAIL table. Never trains (training: train.py on GPU).

    python run.py           # judge results/eval.json, re-check a sample on CPU, write results/
    python run.py --quick   # same with a smaller CPU re-check (< 30 s); what CI runs
    python run.py --clean   # delete caches, smoke outputs and intermediate checkpoints
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

import criteria as C
import model_np
from baselines import KeywordAgent, RandomAgent
from pocketos import make_task, run_episode, split_seeds

HERE = Path(__file__).parent
CKPT, RESULTS = HERE / "checkpoints", HERE / "results"
SPLITS = ["id", "unseen_apps", "long", "wording"]
TITLE = "experiment 01"   # experiment 02 reuses this file with its own folders and title


def peak_ram_gb():
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / (2 ** 30 if sys.platform == "darwin" else 2 ** 20)
    except ImportError:  # Windows: peak working set via K32GetProcessMemoryInfo
        try:
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            k32.K32GetProcessMemoryInfo.restype = wintypes.BOOL
            c = PMC()
            c.cb = ctypes.sizeof(PMC)
            if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
                return float("nan")
            return c.PeakWorkingSetSize / 2 ** 30
        except Exception:
            return float("nan")


def ms_per_decision(n=40):
    """CPU latency of the largest cue model (trained if present, else same-shape random weights)."""
    p = CKPT / "cue-5m.npz"
    cfg, P = model_np.load(p) if p.exists() else (dict(d=256, layers=6, heads=4, loops=1),
                                                    model_np.random_params(dict(d=256, layers=6)))
    agent = model_np.CueAgent(cfg, P)
    phone, task = make_task(split_seeds("id", 1)[0])
    agent(phone, task)  # warm-up
    t = time.perf_counter()
    for _ in range(n):
        agent(phone, task)
    return (time.perf_counter() - t) / n * 1000


def spot_check(n):
    """Re-score baselines (and the saved headline model, if present) on CPU."""
    out = {"keyword": {}, "random": {}}
    for sp in SPLITS:
        out["keyword"][sp] = float(np.mean([run_episode(KeywordAgent(0), s, sp)[0] for s in split_seeds(sp, n)]))
        out["random"][sp] = float(np.mean([run_episode(RandomAgent(0), s, sp)[0] for s in split_seeds(sp, n)]))
    p = CKPT / "cue-2m-s0.npz"
    if p.exists():
        cfg, P = model_np.load(p)
        out["cue-2m-s0"] = {sp: float(np.mean([run_episode(model_np.CueAgent(cfg, P), s, sp)[0]
                                                for s in split_seeds(sp, n)])) for sp in SPLITS}
    return out


def plots(ev, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s, runs = ev["success"], ev["runs"]
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for fam, color, label in (("cue", "#1f6feb", "cue-and-pointer (no text)"), ("text", "#8b949e", "text-token model")):
        pts = {}
        for name, info in runs.items():
            if name.startswith(fam) and not any(t in name for t in ("nomatch", "demos", "rec")):
                key = info["params"]
                pts.setdefault(key, []).append(s[name]["unseen_apps"])
        xs = sorted(pts)
        ax.plot(xs, [100 * np.mean(pts[x]) for x in xs], "o-", color=color, label=label, lw=2)
    ax.axhline(100 * s["keyword"]["unseen_apps"], color="#8b949e", ls="--", lw=1, label="keyword baseline")
    ax.axhline(100 * C.C3_UNSEEN, color="#d29922", ls=":", lw=1.5, label="C3 bar (70%)")
    ax.set_xscale("log")
    ax.set_xlabel("parameters (log scale)", fontsize=11)
    ax.set_ylabel("task success on unseen apps (%)", fontsize=11)
    ax.set_ylim(0, 100)
    um = np.mean([s[n]["unseen_apps"] for n in C.HEADLINE_CUE if n in s])
    ax.set_title(f"Unseen apps: 2M cue model {um:.0%} vs text model "
                 f"{np.mean([s[n]['unseen_apps'] for n in C.HEADLINE_TEXT if n in s]):.0%}", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path / "success_vs_size.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    groups = [("cue-2m", C.HEADLINE_CUE, "#1f6feb"), ("text-2m", C.HEADLINE_TEXT, "#8b949e"),
              ("keyword", ["keyword"], "#d0d7de")]
    w = 0.26
    for i, (label, names, color) in enumerate(groups):
        vals = [100 * np.mean([s[n][sp] for n in names if n in s]) for sp in SPLITS]
        ax.bar(np.arange(len(SPLITS)) + (i - 1) * w, vals, w, label=label, color=color)
    ax.set_xticks(range(len(SPLITS)), ["in-distribution", "unseen apps", "longer tasks", "new wording"], fontsize=10)
    ax.set_ylabel("task success (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_title("Where each model holds up", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path / "splits.png")
    plt.close(fig)


def table(rows):
    lines = ["| # | Criterion | Value | Threshold | Result |", "| --- | --- | --- | --- | --- |"]
    for r in rows:
        res = ("FAIL as expected" if r["ok"] else "PASSED: test can't tell") if r["id"] == "NC" else \
            ("PASS" if r["ok"] else "FAIL")
        lines.append(f"| {r['id']} | {r['name']} | {r['value']} | {r['threshold']} | {res} |")
    return "\n".join(lines)


def clean():
    freed = 0
    targets = [HERE / "__pycache__", HERE / ".pytest_cache", CKPT / "cue-2m-s0-bconly.npz"]
    for t in targets:
        if t.is_dir():
            freed += sum(f.stat().st_size for f in t.rglob("*") if f.is_file())
            shutil.rmtree(t)
        elif t.exists():
            freed += t.stat().st_size
            t.unlink()
    print(f"freed {freed / 2**20:.1f} MB (kept checkpoints/ and results/; delete the repo .venv by hand when done)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()
    if args.clean:
        return clean()
    t0 = time.time()
    RESULTS.mkdir(exist_ok=True)
    n = 20 if args.quick else 100
    check = spot_check(n)
    ms = ms_per_decision(10 if args.quick else 40)
    ev_path = RESULTS / "eval.json"
    if not ev_path.exists():
        print(f"No results/eval.json yet: train on the GPU first (python train.py).\n"
              f"CPU check on {n} tasks/split: keyword {check['keyword']}, random {check['random']}\n"
              f"5M-model decision latency on this CPU: {ms:.1f} ms")
        (RESULTS / "verdict.json").write_text(json.dumps({"verdict": "PENDING", "criteria_passed": "0/5"}, indent=1))
        return
    ev = json.loads(ev_path.read_text())
    kept = sum(ev["checkpoint_bytes"].values()) - ev["checkpoint_bytes"].get("cue-2m-s0-bconly.npz", 0)
    measured = dict(kept_mb=kept / 2 ** 20, ram_gb=peak_ram_gb(), ms_per_decision=ms)
    rows, neg, verdict = C.judge(ev, measured)
    passed = f"{sum(r['ok'] for r in rows)}/{len(rows)}"
    print(table(rows + [neg]))
    print(f"\nVerdict: {verdict} ({passed} criteria)")
    if "cue-2m-s0" in check:
        print(f"CPU re-check of cue-2m-s0 on {n} tasks/split: {check['cue-2m-s0']} "
              f"(GPU eval: { {sp: ev['success']['cue-2m-s0'][sp] for sp in SPLITS} })")
    plots(ev, RESULTS)
    s = ev["success"]
    other = [f"| {k} | " + " | ".join(f"{100 * v[sp]:.1f}" if sp in v else "-" for sp in SPLITS) + " |"
             for k, v in s.items()]
    md = [f"# Results: {TITLE}\n",
          f"seed {args.seed} · quick {args.quick} · GPU {ev.get('gpu')} · torch {ev.get('torch')} · "
          f"{ev.get('n_eval')} tasks per split · GPU training {ev.get('train_minutes_total')} min · "
          f"python {sys.version.split()[0]} · numpy {np.__version__} · peak RAM {measured['ram_gb']:.2f} GB · "
          f"run.py {time.time() - t0:.0f} s\n",
          f"**Verdict: {verdict}** ({passed} criteria)\n", table(rows + [neg]), "",
          "## Task success by model and split (%)\n",
          "| Model | in-distribution | unseen apps | longer tasks | new wording |", "| --- | --- | --- | --- | --- |",
          *other, "", "## Training runs\n", "| Run | Params | Steps | GPU seconds | Bytes |", "| --- | --- | --- | --- | --- |",
          *[f"| {k} | {v['params']:,} | {v['steps']} | {v['train_seconds']} | {v['bytes']:,} |" for k, v in ev["runs"].items()],
          "", "![success vs size](success_vs_size.png)", "", "![splits](splits.png)"]
    (RESULTS / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (RESULTS / "verdict.json").write_text(json.dumps(
        {"verdict": verdict, "criteria_passed": passed, "criteria": rows, "negative_control": neg,
         "cpu_check": check}, indent=1, default=str))


if __name__ == "__main__":
    main()
