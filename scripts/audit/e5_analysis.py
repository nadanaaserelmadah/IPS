"""Audit paper, E5: VS-VIO retrained end to end under the clean protocol, three seeds (scripts/kaggle_kernels/e5_vsvio).

Inputs:
  E5_TXT       scripts/audit/score_released.py on the epochs that scripts/audit/e5_select.py picked
               (rows <seed>/<rule>/poses, rules best08 / besttest / last);
  RELEASED_TXT the E1-E3 output (released checkpoints, incl. vsvio/policy);
  SEEDS_TXT    the E4 output (VIFT, base_s* and fix_s*);
  E5_DIR ...   the kernels' e5/ folders (evals.jsonl: VS-VIO's own t_rel for every epoch 81-99).
Prints the picks, the per-rule means, how much choosing the epoch on the test drives lowers the number (the
authors' train.py keeps best_<t_rel>.pth on val_seq 05 07 10), and the E6 spread with VS-VIO seeds added. Draws
the epoch curves and the per-rule strips.
  python scripts/audit/e5_analysis.py E5_TXT RELEASED_TXT SEEDS_TXT OUT_PNG E5_DIR [E5_DIR ...]
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from e6_spread import AXIS, GRID, INK, INK2, MUTED, NAMES, STAGES, SURFACE, gate_rows  # noqa: E402

RULES = ("best08", "besttest", "last")
RULE_LABEL = {"best08": "best on 08\n(clean)", "besttest": "best on test\n(authors' rule)", "last": "last epoch\n(99)"}
SEED_COLORS = ("#2a78d6", "#1baf7a", "#eb6834")  # validated: scripts/validate_palette.js --pairs all (E6 figure)
RELEASED_KEY = "vsvio/policy"


def rows_08(path):
    """{model: t_rel} from the '## 08' table, network only, OXTS."""
    text = Path(path).read_text()
    if "\n## 08" not in text:
        return {}
    out = {}
    for line in text.split("\n## 08")[1].splitlines():
        m = re.match(r"\| (\S+) \| ([\d.]+) \| ([\d.]+) \|", line)
        if m:
            out[m[1]] = float(m[2])
    return out


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def fmt_range(x):
    return f"{x.min():.2f}–{x.max():.2f} ({np.ptp(x):.2f})"


def main():
    e5_txt, released_txt, seeds_txt, out_png = sys.argv[1:5]
    runs = [Path(p) for p in sys.argv[5:]]

    # ---- scored picks: {seed: {rule: [network, gyro, accel, gate]}}
    scored, held08 = {}, {}
    for key, v in gate_rows(e5_txt).items():
        seed, rule = key.split("/")[:2]
        scored.setdefault(seed, {})[rule] = v
    for key, t in rows_08(e5_txt).items():
        seed, rule = key.split("/")[:2]
        held08.setdefault(seed, {})[rule] = t
    seeds = sorted(scored, key=lambda s: int(s[1:]))
    assert seeds and all(set(scored[s]) == set(RULES) for s in seeds), {s: list(scored[s]) for s in scored}

    # ---- per-epoch history (VS-VIO's own metric, from the kernel)
    hist = {}
    for run in runs:
        st = json.loads((run / "status.json").read_text())
        assert st.get("done") and not st.get("smoke"), f"{run}: not a finished full run"
        ev = [json.loads(line) for line in open(run / "evals.jsonl")]
        assert [e["epoch"] for e in ev] == list(range(81, 100)), run
        hist[f"s{st['seed']}"] = ev
    assert sorted(hist) == sorted(seeds), (sorted(hist), seeds)
    pick = {s: {"best08": min(hist[s], key=lambda e: e["08"]["t_rel"]),
                "besttest": min(hist[s], key=lambda e: e["test_mean_t_rel"]),
                "last": hist[s][-1]} for s in seeds}

    rel = gate_rows(released_txt)
    rel_vsvio = rel[RELEASED_KEY]
    rel5 = np.array([v for k, v in rel.items() if k in NAMES])
    vift = np.array([v for k, v in gate_rows(seeds_txt).items() if k.startswith("base_")])
    assert len(rel5) == 5 and len(vift) == 6

    print("## E5: VS-VIO retrained under the clean protocol (train 00 01 02 04 06 09, 08 held out), "
          f"{len(seeds)} seeds\n")
    print("### E5.1 Epoch picked by each rule, scored like the released models (t_rel %, mean of 05/07/10, OXTS)\n")
    print("| seed | rule | epoch | 08 (held out), network | network | gyro rot + net t | + accel length | gate v3 |")
    print("|---|---|---|---|---|---|---|---|")
    for s in seeds:
        for r in RULES:
            v = scored[s][r]
            h08 = held08.get(s, {}).get(r)
            print(f"| {s} | {r} | {pick[s][r]['epoch']} | {h08:.2f} | " if h08 is not None else
                  f"| {s} | {r} | {pick[s][r]['epoch']} | – | ", end="")
            print(" | ".join(f"{x:.2f}" for x in v[:3]) + f" | **{v[3]:.2f}** |")

    print("\n### E5.2 Mean ± SD over seeds\n")
    print("| rule | " + " | ".join(STAGES) + " |\n|---|" + "---|" * len(STAGES))
    for r in RULES:
        X = np.array([scored[s][r] for s in seeds])
        sd = X.std(axis=0, ddof=1) if len(seeds) > 1 else np.zeros(4)
        print(f"| {r} | " + " | ".join(f"{m:.2f} ± {d:.2f}" for m, d in zip(X.mean(0), sd)) + " |")
    print(f"| released checkpoint (trained on 08, test-selected) | " + " | ".join(f"{x:.2f}" for x in rel_vsvio) + " |")

    print("\n### E5.3 What choosing the epoch on the test drives does\n")
    print("VS-VIO's own t_rel (its `kitti_eval`), all 19 evaluated epochs per seed. 'Optimism' = median epoch minus the "
          "picked epoch, on the test drives: how much lower the reported number is than a typical epoch.\n")
    print("| seed | test t_rel over epochs 81–99: min–max (median) | best-on-08 epoch: test | best-on-test epoch: test | "
          "optimism of best-on-test | optimism of best-on-08 | Spearman(08, test) over epochs |")
    print("|---|---|---|---|---|---|---|")
    opt_t, opt_8 = [], []
    for s in seeds:
        tt = np.array([e["test_mean_t_rel"] for e in hist[s]])
        t8 = np.array([e["08"]["t_rel"] for e in hist[s]])
        med = float(np.median(tt))
        bt, b8 = pick[s]["besttest"]["test_mean_t_rel"], pick[s]["best08"]["test_mean_t_rel"]
        opt_t.append(med - bt)
        opt_8.append(med - b8)
        print(f"| {s} | {tt.min():.2f}–{tt.max():.2f} ({med:.2f}) | {b8:.2f} (ep {pick[s]['best08']['epoch']}) | "
              f"{bt:.2f} (ep {pick[s]['besttest']['epoch']}) | {med - bt:.2f} | {med - b8:+.2f} | {spearman(t8, tt):+.2f} |")
    print(f"| mean | | | | {np.mean(opt_t):.2f} | {np.mean(opt_8):+.2f} | |")
    print("\nSame contrast after scoring (best-on-test minus best-on-08; negative = the test-selected epoch looks better):\n")
    print("| seed | " + " | ".join(STAGES) + " |\n|---|" + "---|" * len(STAGES))
    D = np.array([np.subtract(scored[s]["besttest"], scored[s]["best08"]) for s in seeds])
    for s, d in zip(seeds, D):
        print(f"| {s} | " + " | ".join(f"{x:+.2f}" for x in d) + " |")
    print("| mean | " + " | ".join(f"{x:+.2f}" for x in D.mean(0)) + " |")

    print("\n### E5.4 E6 again, with VS-VIO seeds (clean, best on 08)\n")
    V = np.array([scored[s]["best08"] for s in seeds])
    print("| stage | 5 released checkpoints: min–max (range) | VIFT, 6 seeds: min–max (range), SD | "
          f"VS-VIO, {len(seeds)} seeds: min–max (range), SD |\n|---|---|---|---|")
    for i, st in enumerate(STAGES):
        vsd = V[:, i].std(ddof=1) if len(seeds) > 1 else 0.0
        print(f"| {st} | {fmt_range(rel5[:, i])} | {fmt_range(vift[:, i])}, {vift[:, i].std(ddof=1):.2f} | "
              f"{fmt_range(V[:, i])}, {vsd:.2f} |")
    allv = np.array([scored[s][r] for s in seeds for r in RULES])
    print(f"\nReleased VS-VIO checkpoint: network {rel_vsvio[0]:.2f}, gate v3 {rel_vsvio[3]:.2f}. Clean seeds (best on 08) "
          f"with a worse network: {int(np.sum(V[:, 0] > rel_vsvio[0]))} of {len(seeds)}; any rule: "
          f"{int(np.sum(allv[:, 0] > rel_vsvio[0]))} of {len(allv)}. Worse after gate v3: "
          f"{int(np.sum(V[:, 3] > rel_vsvio[3]))} of {len(seeds)}.")

    draw(seeds, hist, pick, scored, rel_vsvio, out_png)


def draw(seeds, hist, pick, scored, rel_vsvio, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": AXIS,
                         "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.5, 1, 1]})
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    ax = axes[0]  # epoch curves, VS-VIO's own metric
    for s, c in zip(seeds, SEED_COLORS):
        ep = [e["epoch"] for e in hist[s]]
        tt = [e["test_mean_t_rel"] for e in hist[s]]
        ax.plot(ep, tt, color=c, lw=2, zorder=2)
        ax.annotate(s, (ep[-1], tt[-1]), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8, color=INK2)
        b8, bt = pick[s]["best08"], pick[s]["besttest"]
        ax.scatter([b8["epoch"]], [b8["test_mean_t_rel"]], s=70, facecolors=SURFACE, edgecolors=c, linewidths=2, zorder=4)
        ax.scatter([bt["epoch"]], [bt["test_mean_t_rel"]], s=60, c=c, marker="D", edgecolors=SURFACE, linewidths=1.5,
                   zorder=5)
    ax.axhline(rel_vsvio[0], color=MUTED, lw=1, ls=(0, (4, 3)))
    ax.annotate("released checkpoint", (81, rel_vsvio[0]), xytext=(0, 4), textcoords="offset points", fontsize=7,
                color=MUTED)
    ax.set_xlim(80.5, 100.8)
    ax.set_xticks([81, 85, 90, 95, 99])
    allt = [e["test_mean_t_rel"] for s in seeds for e in hist[s]] + [rel_vsvio[0]]
    lo, hi = min(allt), max(allt)
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.22 * (hi - lo))  # a band on top for the one-row legend
    ax.set_xlabel("epoch")
    ax.set_ylabel("t_rel % (KITTI 05/07/10)")
    ax.set_title("Test error per epoch (VS-VIO's own metric)", color=INK, fontsize=10, loc="left")
    ax.legend(handles=[Line2D([], [], color=c, lw=2, label=s) for s, c in zip(seeds, SEED_COLORS)] +
              [Line2D([], [], ls="", marker="o", mfc=SURFACE, mec=INK2, mew=1.5, ms=7, label="picked on 08"),
               Line2D([], [], ls="", marker="D", color=INK2, ms=6, label="picked on test")],
              fontsize=7, frameon=False, loc="upper center", ncol=5, handlelength=1.6, columnspacing=1.2)

    for ax, col, title in ((axes[1], 0, "Network output"), (axes[2], 3, "After gyro + accel + gate v3")):
        for gx, r in enumerate(RULES):
            y = np.array([scored[s][r][col] for s in seeds])
            ax.plot([gx - 0.28, gx + 0.28], [y.mean()] * 2, color=AXIS, lw=1.5, zorder=1, solid_capstyle="round")
            for j, (s, c) in enumerate(zip(seeds, SEED_COLORS)):
                ax.scatter([gx + (j - 1) * 0.12], [y[j]], s=46, c=c, edgecolors=SURFACE, linewidths=1.5, zorder=3)
        ax.axhline(rel_vsvio[col], color=MUTED, lw=1, ls=(0, (4, 3)))
        ax.annotate("released", (2.45, rel_vsvio[col]), xytext=(0, 4), textcoords="offset points", ha="right",
                    fontsize=7, color=MUTED)
        ax.set_xticks(range(3), [RULE_LABEL[r] for r in RULES], color=INK2, fontsize=7.5)
        ax.set_xlim(-0.5, 2.5)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    axes[1].set_ylabel("t_rel % (05/07/10, OXTS)")
    fig.text(0.01, 0.01, "Bars: mean over seeds. Clean protocol: trained without 08; 'best on 08' never looks at "
             "05/07/10. Released checkpoint: trained with 08, kept by its test score.", fontsize=7, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_png, dpi=200, facecolor=SURFACE)
    fig.savefig(Path(out_png).with_suffix(".pdf"), facecolor=SURFACE)
    print(f"\nfigure: {out_png}")


if __name__ == "__main__":
    main()
