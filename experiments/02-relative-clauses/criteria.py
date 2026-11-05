"""Pre-registered hypothesis, pass criteria, controls and budgets for experiment 02.

Written 2026-11-05, before any experiment-02 training. Same thresholds as experiment 01 (agreed with
Obi on 2026-11-04); only the model input changed. Thresholds never move after results.

Ablation (reported, not a criterion): control_clause_blind = the cue-2m-s0 model with every cue shown
as "current clause". If the relative signal is what fixes long tasks, its "long" score drops well below
the real model's.
"""

GOAL = ("Fix experiment 01's long-task failure: tell the model which instruction clause it is on, "
        "store models in float16, and keep generalising to unseen apps.")
HYPOTHESIS = ("With clause ids relative to the current (first unfinished) clause, the 2M cue-and-pointer "
              "policy completes >= 60% of tasks with 4-6 sub-goals (01: 5%) while still completing >= 70% "
              "of tasks in unseen apps, and all kept float16 models fit in 50 MB (01: 70 MB float32).")

# Budgets (C1)
MAX_KEPT_MB = 50.0          # all kept checkpoints together
MAX_TRAIN_MIN = 10.0        # GPU training, all runs together (Obi: ~5 min, never more than 10)
MAX_RAM_GB = 2.0            # evaluation process
MAX_MS_PER_DECISION = 10.0  # largest cue model, NumPy on CPU

# Thresholds. Headline model = cue-2m, mean of 3 training seeds.
C2_ID = 0.90                # FunctionGemma reached 85% held-out after fine-tuning; we ask for more in-distribution
C3_UNSEEN = 0.70
C3_MARGIN_OVER_KEYWORD = 0.25
C4_LONG = 0.60
C5_MARGIN_OVER_TEXT = 0.15  # cue-2m minus text-2m on unseen apps
NEG_CONTROL_MAX = 0.15      # blind-prompt control must stay below this (else INCONCLUSIVE)

CORE = "C3"                 # the core criterion: if it fails, the verdict is FAIL
HEADLINE_CUE = ["cue-2m-s0", "cue-2m-s1", "cue-2m-s2"]
HEADLINE_TEXT = ["text-2m-s0"]   # one seed: all three 01 seeds scored 0.0%


def _mean(success, names, split):
    vals = [success[n][split] for n in names if n in success]
    if not vals:
        return None, None
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)) ** 0.5 if len(vals) > 1 else 0.0
    return m, sd


def judge(eval_json, measured):
    """Return (rows, neg_control_row, verdict). rows: list of dicts criterion/value/threshold/pass."""
    s = eval_json["success"]
    rows = []
    kept_mb = measured["kept_mb"]
    c1_ok = (kept_mb <= MAX_KEPT_MB and eval_json.get("train_minutes_total", 1e9) <= MAX_TRAIN_MIN
             and measured["ram_gb"] <= MAX_RAM_GB and measured["ms_per_decision"] <= MAX_MS_PER_DECISION)
    rows.append(dict(id="C1", name="Budget: kept models, GPU training time, eval RAM, CPU ms/decision (5M model)",
                     value=f"{kept_mb:.1f} MB, {eval_json.get('train_minutes_total', float('nan')):.1f} min, "
                           f"{measured['ram_gb']:.2f} GB, {measured['ms_per_decision']:.1f} ms",
                     threshold=f"<= {MAX_KEPT_MB:.0f} MB, <= {MAX_TRAIN_MIN:.0f} min, <= {MAX_RAM_GB:.0f} GB, "
                               f"<= {MAX_MS_PER_DECISION:.0f} ms",
                     ok=c1_ok))
    idm, ids = _mean(s, HEADLINE_CUE, "id")
    rows.append(dict(id="C2", name="Learns the job: cue-2m success, in-distribution",
                     value=f"{idm:.1%} ± {ids:.1%}", threshold=f">= {C2_ID:.0%}", ok=idm >= C2_ID))
    um, us = _mean(s, HEADLINE_CUE, "unseen_apps")
    kw = s["keyword"]["unseen_apps"]
    rows.append(dict(id="C3", name="Generalises to unseen apps: cue-2m success",
                     value=f"{um:.1%} ± {us:.1%} (keyword {kw:.1%})",
                     threshold=f">= {C3_UNSEEN:.0%} and >= keyword + {C3_MARGIN_OVER_KEYWORD * 100:.0f} pts",
                     ok=um >= C3_UNSEEN and um >= kw + C3_MARGIN_OVER_KEYWORD))
    lm, ls = _mean(s, HEADLINE_CUE, "long")
    rows.append(dict(id="C4", name="Chains skills: cue-2m success, tasks with 2x the sub-goals",
                     value=f"{lm:.1%} ± {ls:.1%}", threshold=f">= {C4_LONG:.0%}", ok=lm >= C4_LONG))
    tm, tsd = _mean(s, HEADLINE_TEXT, "unseen_apps")
    gap = um - tm
    rows.append(dict(id="C5", name="No language needed: cue-2m minus text-2m, unseen apps",
                     value=f"{gap * 100:+.1f} pts (text {tm:.1%} ± {tsd:.1%})",
                     threshold=f">= +{C5_MARGIN_OVER_TEXT * 100:.0f} pts", ok=gap >= C5_MARGIN_OVER_TEXT))
    neg = s["control_blind_prompt"]["unseen_apps"]
    neg_row = dict(id="NC", name="Negative control: cue-2m given another task's instruction, unseen apps",
                   value=f"{neg:.1%}", threshold=f"must stay < {NEG_CONTROL_MAX:.0%}", ok=neg < NEG_CONTROL_MAX)
    passed = sum(r["ok"] for r in rows)
    if not neg_row["ok"]:
        verdict = "INCONCLUSIVE"
    elif passed == len(rows):
        verdict = "PASS"
    elif not next(r for r in rows if r["id"] == CORE)["ok"]:
        verdict = "FAIL"
    else:
        verdict = "INCONCLUSIVE"
    return rows, neg_row, verdict
