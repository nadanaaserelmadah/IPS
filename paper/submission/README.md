# Short paper (draft; venue not decided)

`main.tex` is the short IEEE two-column version (currently 8 pages including references: the 6-page short version plus the PI-VIFT description and figure, the accelerometer-alignment equations, the design history of the gate with a sensitivity table, two more limitations, and the cross-model results on the released UL-VIO and NASVIO weights; the numbers come from `results_cache/2026-11-09/audit_E1_E3_released_models.txt`, the same models are the subject of the separate audit paper in `paper/audit/`). It does not name a venue; that is decided after talking with the supervisors. `refs.bib` holds the bibliography and `figures/` the figures. `IEEEtran.cls` and `IEEEtran.bst` are local copies from CTAN, because the system TeX (BasicTeX) does not include them.

Build:

```bash
export PATH=/Library/TeX/texbin:$PATH
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Every number is checked against `docs/RESULTS.md`; the last check was 2026-11-01, by an independent pass over 88 claims, with all fixes applied. The plan and the list of material moved to the supplementary are in `docs/SUBMISSION_PLAN.md`.

`versions/` keeps earlier copies of `main.tex` (and the PDF built from each): `main_v1_old_intro.*` is the version before the introduction was rewritten (2026-11-15), with the original opening paragraph that starts from the learned-VIO literature.
