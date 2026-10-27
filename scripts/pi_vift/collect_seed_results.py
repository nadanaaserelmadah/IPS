"""Aggregate test metrics over seed repeats into a markdown table (mean ± std per arm).

Each run dir holds test_metrics_<tag>.csv (tag = best | bestate | last) and status.json
as written by the Kaggle kernels (pivift_qfix*, vift_qfix*). A tag whose status says
"same as <other>" (same epoch, test skipped) reuses <other>'s metrics. Extra test
sequences from the eval_ext kernel go in test_metrics_<tag>_ext.csv and are merged in;
"SE3 mean (all)" then averages every sequence present.

  python scripts/pi_vift/collect_seed_results.py \\
      --arm "PI-VIFT=logs/qfix/pivift_s42,logs/qfix/pivift_s7,logs/qfix/pivift_s123" \\
      --arm "VIFT-style=logs/qfix/vift_s42,logs/qfix/vift_s7,logs/qfix/vift_s123"
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

SEQS = ("MH_05_difficult", "V2_02_medium")
EXT_SEQS = ("V1_01_easy", "V1_02_medium")
TAGS = ("best", "bestate", "last")
TAG_NAMES = {"best": "best val/loss", "bestate": "best val/ate_se3", "last": "last"}


def load_run(run: Path) -> dict[str, dict[str, float]]:
    """{tag: {metric: value}} for every tag tested (or aliased) in this run."""
    status = json.loads((run / "status.json").read_text()) if (run / "status.json").is_file() else {}
    out: dict[str, dict[str, float]] = {}
    for tag in TAGS:
        path = run / f"test_metrics_{tag}.csv"
        if path.is_file():
            out[tag] = {}
            for part in (path, run / f"test_metrics_{tag}_ext.csv"):
                if part.is_file():
                    row = next(csv.DictReader(part.open()))
                    out[tag].update({k.removeprefix("test/"): float(v) for k, v in row.items()
                                     if k.startswith("test/") and v and not k.startswith(("test/mean", "test/n_"))})
            out[tag]["epoch"] = float(status.get(f"{tag}_epochs", "nan"))
    for tag in TAGS:
        alias = str(status.get(f"test_{tag}", ""))
        if tag not in out and alias.startswith("same as "):
            src = alias.removeprefix("same as ").strip()
            if src in out:
                out[tag] = dict(out[src], epoch=float(status.get(f"{tag}_epochs", out[src]["epoch"])))
    return out


def fmt(vals: list[float], digits: int = 2) -> str:
    v = np.asarray(vals, dtype=np.float64)
    if len(v) == 1:
        return f"{v[0]:.{digits}f}"
    return f"{v.mean():.{digits}f} ± {v.std(ddof=1):.{digits}f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arm", action="append", required=True, help="NAME=dir1,dir2,...")
    args = p.parse_args()

    print("| Arm | ckpt | seeds | **SE3 mean** | SE3 mean (all seqs) | SE3 MH_05 | SE3 V2_02 | SE3 V1_01 | SE3 V1_02 "
          "| Sim3 MH_05 / V2_02 | sim3_scale MH_05 / V2_02 | est path MH_05 / V2_02 | epochs | per-seed SE3 mean |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for spec in args.arm:
        name, dirs = spec.split("=", 1)
        runs = [load_run(Path(d)) for d in dirs.split(",") if d]
        for tag in TAGS:
            rows = [r[tag] for r in runs if tag in r]
            if not rows:
                continue
            g = lambda key: [r[key] for r in rows]  # noqa: E731
            se3 = [(r[f"{SEQS[0]}_ate_se3_m"] + r[f"{SEQS[1]}_ate_se3_m"]) / 2 for r in rows]
            se3_all = [np.mean([v for k, v in r.items() if k.endswith("_ate_se3_m")]) for r in rows]
            n_seq = {sum(k.endswith("_ate_se3_m") for k in r) for r in rows}
            ext = lambda s: fmt(g(f"{s}_ate_se3_m")) if all(f"{s}_ate_se3_m" in r for r in rows) else "—"  # noqa: E731
            print(
                f"| {name} | {TAG_NAMES[tag]} | {len(rows)} | **{fmt(se3)}** "
                f"| {fmt(se3_all) if n_seq != {2} else '—'} ({'/'.join(map(str, sorted(n_seq)))} seqs) "
                f"| {fmt(g(f'{SEQS[0]}_ate_se3_m'))} | {fmt(g(f'{SEQS[1]}_ate_se3_m'))} "
                f"| {ext(EXT_SEQS[0])} | {ext(EXT_SEQS[1])} "
                f"| {fmt(g(f'{SEQS[0]}_ate_sim3_m'))} / {fmt(g(f'{SEQS[1]}_ate_sim3_m'))} "
                f"| {fmt(g(f'{SEQS[0]}_sim3_scale'))} / {fmt(g(f'{SEQS[1]}_sim3_scale'))} "
                f"| {fmt(g(f'{SEQS[0]}_est_path_len_m'), 0)} / {fmt(g(f'{SEQS[1]}_est_path_len_m'), 0)} "
                f"| {', '.join(str(int(e)) if e == e else '?' for e in g('epoch'))} "
                f"| {', '.join(f'{x:.2f}' for x in se3)} |"
            )


if __name__ == "__main__":
    main()
