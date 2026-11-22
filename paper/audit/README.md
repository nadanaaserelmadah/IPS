# Audit paper: first draft

`main.tex` is the first full draft (2026-11-14) of the second paper in the project: *Benchmarking Learned Visual–Inertial Odometry on KITTI: An Audit of Data, Protocol, and Evaluation*. The plan behind it is in `docs/AUDIT_PAPER_PLAN.md`, and its numbers come from sections 4.1–4.5 there and from `results_cache/2026-11-09`, `2026-11-10`, `2026-11-12` and `2026-11-14`.

## Build

```bash
cd paper/audit && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`\ttdefault` is set to `cmtt` because the Courier fonts that IEEEtran expects are not installed on this machine.

Figures are copied from `docs/figures`:
- `audit_e6_spread.pdf` comes from `scripts/audit/e6_spread.py`.
- `audit_e5.pdf` comes from `scripts/audit/e5_analysis.py`.
- `gyro_timing.png` is shared with the first paper.

## Open before submission

These are also marked in red with `\todo{}` in the PDF, or as `TODO` in `refs.bib`.

1. **TOR-VIO and Gu et al. 2026** (paywalled) must be read before claiming that the timing error was not reported before (Sec. IV). So far: Gu's abstract and reference list (Scopus export, `sources/Gu2026_scopus_export.csv`, 2026-11-21) show a unified benchmark of geometric, hybrid and learned VIO on EuRoC, TUM VI, KITTI and FusionPortableV2 whose only learned VIO reference is VS-VIO, with nothing on inertial timing or training protocols. The full text is still needed. TOR-VIO: nothing read yet.
2. **Paper numbers** (checked 2026-11-18):
   - VS-VIO: the paper's 2.40±0.064 (λ=3e-5, 10 seeds) matches our 2.35 for the released weights.
   - UL-VIO: the paper has no t_rel table, only RMSE and an unlabelled figure. Its README example output has a mean of 3.52, identical to ours.
   - **NASVIO:** the paper is not openly accessible. Compare its printed numbers before submission.
3. **Shared data in other methods:**
   - RWKV-VIO (checked 2026-11-18): its code uses VS-VIO's loader and `imus/<seq>.mat` format, trains on 08, and keeps `best_<t_rel>.pth` on 05/07/10. Now in Table I. The `.mat` files are not in the repo, so there is no MD5 check.
   - DB-VIO's paper interpolates the 100 Hz stream to image timestamps with 11 samples per pair and integrates the gyroscope with an unstated step. Its code is needed.
   - DAP-Pose: unchecked, no code.
4. **Bibliography:** filled from Crossref on 2026-11-18 (NASVIO, UL-VIO, RWKV-VIO, TOR-VIO, Gu 2026). Nothing left marked TODO in `refs.bib`.
5. **Overlap with the first paper:** the timing finding and the VIFT protocol audit also appear in `paper/submission/main.tex`. Decide which paper keeps the full account. The plan proposes this one, with a line and a citation in the first.
6. **Author line:** it is copied from the first paper. Set it as intended.
