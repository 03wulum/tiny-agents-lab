"""GPU training + evaluation for experiment 02. Runs on Fortran's NVIDIA GPU only.

    python train.py            # full sweep, ~4 min of training (hard cap 5.5), then GPU evaluation
    python train.py --smoke    # 1-minute end-to-end check
    python train.py --eval-only

Same pipeline as experiment 01 (BC + 3 DAgger rounds), with three changes:
relative clause ids, the fixed "and then" parser, and float16 checkpoints.
"""
from _shared import HERE, load01

T = load01("train")
T.CKPT, T.RESULTS = HERE / "checkpoints", HERE / "results"
T.RELATIVE = True
T.SAVE_DTYPE = "float16"
R = {"relative": True}
# name: (family, size, train seed, seconds, n demo episodes, extra). Same sizes as 01, more time for the
# 2M headline seeds (C2 was 85% after 20 s), one text rival seed (it scored 0% on all three seeds in 01).
T.RUNS = [
    ("cue-20k", "cue", "20k", 0, 10, 20000, R),
    ("cue-100k", "cue", "100k", 0, 10, 20000, R),
    ("cue-500k", "cue", "500k", 0, 15, 20000, R),
    ("cue-2m-s0", "cue", "2m", 0, 30, 20000, R),
    ("cue-2m-s1", "cue", "2m", 1, 30, 20000, R),
    ("cue-2m-s2", "cue", "2m", 2, 30, 20000, R),
    ("cue-5m", "cue", "5m", 0, 25, 20000, R),
    ("text-2m-s0", "text", "2m", 0, 30, 20000, {}),
]

if __name__ == "__main__":
    T.main()
