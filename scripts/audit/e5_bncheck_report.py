"""Audit paper, E5 check: summarise scripts/kaggle_kernels/e5_bncheck (BatchNorm statistics re-estimated, no weight
changed) for the E5 best-on-08 checkpoints and the released VS-VIO checkpoint.
  python scripts/audit/e5_bncheck_report.py BN_DIR SCORED_TXT
BN_DIR: the kernel's bn/ folder (results.jsonl, bn_shift.json). SCORED_TXT: scripts/audit/score_released.py on
BN_DIR/*/*/k* (rows <model>/<variant>/k<k>).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e6_spread import STAGES, gate_rows  # noqa: E402

MODELS = ("s0_best08", "s1_best08", "s2_best08", "released")
VARIANTS = ("asis", "recal")


def main():
    bn, scored_txt = Path(sys.argv[1]), sys.argv[2]
    res = [json.loads(line) for line in open(bn / "results.jsonl")]
    by = defaultdict(list)
    for r in res:
        by[(r["model"], r["variant"], r["k"])].append(r)
    K = sorted({r["k"] for r in res})

    print(f"## BatchNorm check: statistics re-estimated on the clean training split, batch 16, one GPU ({len(K)} gumbel seeds)\n")
    print("### VS-VIO's own metric (t_rel %, mean of 05/07/10; mean ± SD over gumbel seeds) and forward scale\n")
    print("| model | variant | 05/07/10 | 08 | forward scale 05/07/10 (est / GT) | visual usage % |\n|---|---|---|---|---|---|")
    for m in MODELS:
        for v in VARIANTS:
            test = [np.mean([r["t_rel"] for r in by[(m, v, k)] if r["seq"] != "08"]) for k in K]
            t08 = [r["t_rel"] for k in K for r in by[(m, v, k)] if r["seq"] == "08"]
            sc = [r["fwd_scale"] for k in K for r in by[(m, v, k)] if r["seq"] != "08"]
            us = [r["usage"] for k in K for r in by[(m, v, k)] if r["seq"] != "08"]
            print(f"| {m} | {v} | {np.mean(test):.2f} ± {np.std(test, ddof=1) if len(K) > 1 else 0:.2f} | "
                  f"{np.mean(t08):.2f} | {np.mean(sc):.3f} | {np.mean(us):.0f} |")

    rows = gate_rows(scored_txt)
    print("\n### Scored like every audit model (OXTS, mean over gumbel seeds)\n")
    print("| model | variant | " + " | ".join(STAGES) + " |\n|---|---|" + "---|" * len(STAGES))
    for m in MODELS:
        for v in VARIANTS:
            X = np.array([rows[f"{m}/{v}/k{k}"] for k in K if f"{m}/{v}/k{k}" in rows])
            if len(X):
                print(f"| {m} | {v} | " + " | ".join(f"{x:.2f}" for x in X.mean(0)) + " |")

    shift = json.loads((bn / "bn_shift.json").read_text())
    print("\n### How far the saved statistics were from the re-estimated ones (mean over layers)\n")
    print("| model | image + policy BN: mean shift in SD | IMU encoder BN (BatchNorm1d 64/128/256): mean shift in SD |")
    print("|---|---|---|")
    for m in MODELS:
        L = shift[m]
        imu = [x for x in L if x["kind"] == "BatchNorm1d" and x["features"] in (64, 128, 256)][:3]
        rest = [x for x in L if x not in imu]
        print(f"| {m} | {np.mean([x['mean_shift_in_sd'] for x in rest]):.2f} | {np.mean([x['mean_shift_in_sd'] for x in imu]):.2f} |")
    print("\n(The kernel's per-layer variance ratio is not shown: channels with zero variance after ReLU dominate it.)")


if __name__ == "__main__":
    main()
