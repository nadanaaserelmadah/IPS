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

1. **TOR-VIO and Gu et al. 2026** (paywalled) must be read before claiming that the timing error was not reported before (Sec. III).
2. **The UL-VIO and NASVIO papers' own numbers** still need comparing with their released checkpoints (Sec. VII). UL-VIO's released weights give 3.36 on its training sequence 08, close to its 3.52 on the test sequences.
3. **Shared data in other methods:** we have not confirmed that RWKV-VIO, DB-VIO and DAP-Pose use the same `imus/*.mat`. DB-VIO's paper (read 2026-11-15) says it interpolates the 100 Hz stream to image timestamps with 11 samples per image pair and integrates the gyroscope explicitly (its attitude-guided encoding) with a time step it does not state; whether it uses the same files or the nominal 10 ms step needs its code. Table I lists only the repositories that were checked by MD5.
4. **Bibliography:** RWKV-VIO, TOR-VIO and Gu 2026 have placeholder authors and titles. The NASVIO and UL-VIO author lists need checking.
5. **Authors:** contact the authors of VS-VIO, NASVIO, UL-VIO and VIFT before posting. The paper criticises their protocol, so the tone stays factual.
6. **Overlap with the first paper:** the timing finding and the VIFT protocol audit also appear in `paper/submission/main.tex`. Decide which paper keeps the full account. The plan proposes this one, with a line and a citation in the first.
7. **Author line:** it is copied from the first paper. Set it as intended.
