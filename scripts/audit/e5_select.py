"""Audit paper, E5: pick the epoch of each VS-VIO retraining run (scripts/kaggle_kernels/e5_vsvio) by three rules
and lay the chosen per-epoch dumps out for scripts/audit/score_released.py.

Rules, all over the epochs the authors evaluate (81-99):
  best08  - lowest t_rel on 08 (clean: 08 is held out from training, the test drives are never looked at);
  besttest - lowest mean t_rel on 05/07/10 (what the authors' train.py keeps: best_<t_rel>.pth on val_seq 05 07 10);
  last    - epoch 99.
Prints the selection with VS-VIO's own t_rel per drive, and creates FARM/<run>/<rule>/poses -> <run>/epochs/<ep>.
  python scripts/audit/e5_select.py FARM E5_DIR [E5_DIR ...]      (E5_DIR = a kernel's downloaded e5/ folder)
then
  python scripts/audit/score_released.py FARM/*/*/poses -- GEO NPZ GYRO RAW_CALIB SUMA
"""
import json
import sys
from pathlib import Path

import numpy as np


def main():
    farm, runs = Path(sys.argv[1]), [Path(p) for p in sys.argv[2:]]
    print("| run | rule | epoch | 08 | 05 | 07 | 10 | mean 05/07/10 |\n|---|---|---|---|---|---|---|---|")
    summary = {}
    for run in runs:
        st = json.loads((run / "status.json").read_text())
        assert st.get("done") and not st.get("smoke"), f"{run}: not a finished full run"
        ev = [json.loads(line) for line in open(run / "evals.jsonl")]
        eps = [e["epoch"] for e in ev]
        assert eps == list(range(81, 100)), f"{run}: evaluated epochs {eps}"
        name = f"s{st['seed']}"
        pick = {"best08": min(ev, key=lambda e: e["08"]["t_rel"]),
                "besttest": min(ev, key=lambda e: e["test_mean_t_rel"]),
                "last": ev[-1]}
        for rule, e in pick.items():
            src = (run / "epochs" / f"{e['epoch']:03d}").resolve()
            assert all((src / f"{s}_estimated_poses.npy").is_file() for s in ("05", "07", "10", "08")), src
            dst = farm / name / rule / "poses"
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink():
                dst.unlink()
            dst.symlink_to(src)
            print(f"| {name} | {rule} | {e['epoch']} | {e['08']['t_rel']:.2f} | {e['05']['t_rel']:.2f} | "
                  f"{e['07']['t_rel']:.2f} | {e['10']['t_rel']:.2f} | {e['test_mean_t_rel']:.2f} |")
        summary[name] = {r: pick[r]["test_mean_t_rel"] for r in pick}
    print("\n| rule | mean 05/07/10 per seed | mean ± SD |\n|---|---|---|")
    for rule in ("best08", "besttest", "last"):
        v = np.array([summary[n][rule] for n in summary])
        print(f"| {rule} | {' / '.join(f'{x:.2f}' for x in v)} | {v.mean():.2f} ± {v.std(ddof=1) if len(v) > 1 else 0:.2f} |")


if __name__ == "__main__":
    main()
