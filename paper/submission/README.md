# Short paper (draft; venue not decided)

`main.tex` is the short IEEE two-column version (currently 6 pages including references). It does not name a venue; that is decided after talking with the supervisors. `refs.bib` holds the bibliography and `figures/` the figures. `IEEEtran.cls` and `IEEEtran.bst` are local copies from CTAN, because the system TeX (BasicTeX) does not include them.

Build:

```bash
export PATH=/Library/TeX/texbin:$PATH
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Every number is checked against `docs/RESULTS.md`; the last check was 2026-11-01, by an independent pass over 88 claims, with all fixes applied. The plan and the list of material moved to the supplementary are in `docs/SUBMISSION_PLAN.md`.
