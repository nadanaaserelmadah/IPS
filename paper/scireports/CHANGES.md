# Scientific Reports version: what changed

Source of every number and claim: `paper/submission/main.tex` (IEEE-style, 8 pages), which was not edited. No result was changed or added. The online gyro-bias work is not in the paper (Limitations (vi) says an online estimate is future work, as before).

Status: **built** (`scripts/scireports/build_scireports.sh`): `main.pdf` 17 pages and `supplementary.pdf` 3 pages compile with no errors and no warnings in the logs (no `!` errors, no Overfull/Underfull, no undefined references, no BibTeX warnings). Word counts come from `scripts/scireports/count_words.py` / `wc` on `main.tex` (LaTeX markup stripped, tables, figures and captions excluded).

## Files

| File | Content |
|---|---|
| `main.tex`, `refs.bib` | Manuscript (`\documentclass[pdflatex,sn-nature]{sn-jnl}`), one file, no `\input`, no footnotes. Bibliography verified by a helper (`refs_check.md`). |
| `sn-jnl.cls`, `sn-nature.bst` | Copied from the official Springer Nature template (December 2024), which is unpacked in `template/`. |
| `supplementary.tex` | Supplementary Figs. S1–S3 and Tables S1–S3. |
| `figsrc/`, `figures/` | TikZ sources (Helvetica, 1 pt lines, grey plus one blue) and the exported vector PDFs, one file per figure. |
| `cover_letter.md` | Draft with blanks. |
| `scripts/scireports/` (repo root) | Figure scripts for the two plots, `build_scireports.sh`, `count_words.py`. |

## Counts

| Item | Limit | Now |
|---|---|---|
| Title | 20 words | 14 words |
| Abstract | 200 words, unstructured | 196 words (was about 321) |
| Main text (Introduction, Results, Discussion; captions, tables, Methods excluded) | 4,500 words | 3,338 |
| Methods | no limit | 1,259 |
| Keywords | 6 | 6 |
| Display items in the main paper | 8 | 8 (4 figures, 4 tables) |
| References | about 60 | 34 |

## Structure: what moved where

Old (IEEE) section → new:

- Abstract (cut) → Abstract; the figurative title "Measure What Is Observable, Learn the Rest" → a descriptive 14-word title. The idea "measure what the sensors observe, learn the rest" is in the Introduction and Discussion.
- Introduction + Related Work → **Introduction** (Related Work folded in with its citations; abbreviations defined; "observable", "parallax", "oracle swap" explained).
- Diagnosis → **Results**: "An oracle swap shows which component fails" (Table 1, Fig. 1).
- Method (pipeline, gate) → one Results subsection "An observability gate chooses between measurement and network" (Fig. 2) plus **Methods** (learned branch and PI-VIFT, gyroscope, direction, length with Eq. 1, gate, online trust v4, runtime).
- Experiments (setup and protocol, KITTI, cross-model, EuRoC, TUM-VI) → **Results** subsections; data, networks and metrics → **Methods**.
- Design history and constants → Results (text) + Supplementary Tables S1 and S2.
- Discussion and Limitations + Conclusion → **Discussion** (no subheadings; Limitations (i)–(viii) kept in running text, Conclusion merged in as its first and last paragraphs).
- New statement sections (after Methods, before References, the Springer Nature template's own position): Acknowledgements, Author contributions, Data availability, Code availability, Competing interests, Additional information (ethics: not applicable). **Choice noted:** the guidelines page is ambiguous on where Data availability goes; I followed the template, which puts Declarations before the bibliography. Figure legends and tables stay in the text, where the template puts them (allowed for a first submission).

## Abstract: what was cut

Removed the opening three sentences about gyroscope, accelerometer and two views (condensed to one sentence on the sensors), the list of what the swap exposes for a "recent transformer VIO model", the parenthetical list of gate signals (parallax, solver uncertainty, conditioning, gravity consistency), the 1.65% against lidar poses, the EuRoC 5.5–6.6 m to below 1 m, the "with-08 splits" and "authors' weights" qualifiers, the TUM-VI numbers, and the closing slogan. Kept: oracle-swap diagnosis, 104 ms timing error, observability gate, KITTI 1.96 to 1.56%, cross-model 45–51%, EuRoC/TUM-VI, ORB-SLAM3 about 22 times more accurate, v4 post-hoc, V2_03 (an unseen sequence) not better, every failure reported. The source abstract said 45–59%; 59% was the VS-VIO full-vision case (3.88 to 1.59%), which stays in Results. The abstract now says 45–51%, matching the text.

## Display items (8 in the main paper)

Main paper: Table 1 (oracle swaps), Fig. 1 (timing), Fig. 2 (architecture, redrawn), Table 2 (KITTI), Fig. 3 (KITTI heading error), Table 3 (EuRoC), Table 4 (gate across datasets), Fig. 4 (measure vs learn). This is the suggested selection; it keeps every claim in the abstract backed by a main-paper table or figure.

Supplementary Information (`supplementary.pdf`): Supplementary Fig. S1 (learned branch), S2 (PI-VIFT), S3 (gate decisions); Supplementary Table S1 (four gate versions), S2 (condition-number sensitivity), S3 (comparison with prior work). Each is cited in the text with the word "Supplementary". Table S3's reference numbers point to a short list at the end of that file.

## Other changes

- British spelling (optimisation, normalised, artefact, localisation); no use of "significant"; "±" is stated as standard deviation over 3 seeds (Table 2 and Methods); no P values; no colour in tables.
- Table notes (†) moved into captions; `Hand` spelled out as "Hand-designed" with a definition.
- TikZ figures redrawn: Helvetica, 7 pt, white background, lines at least 1 pt, grey plus one dark blue, bold lowercase panel letters, no fills other than light grey. "Edge" is now "frame pair" in the text.
- Author list kept at six names as in the IEEE version (Rasha Elsayed Sakr included; the task note listed five).
- `refs.bib`: Nature-style fields, full venue names, DOIs and page ranges. Differences from the old file: Wang2020 year 2020 to 2021 (PMLR volume 155), Chen2022 pages conflict between CVF (6646–6655, used) and IEEE (6636–6645), Behley2018 has no pages (paper number only), Clark2017 pages not seen on the AAAI site, LNCS volume numbers missing for Yang2022 and Park2024ULVIO. See `refs_check.md`.

## Open items (only the authors can fill)

See the list at the end of the session report: corresponding author, author contributions, competing interests, funding, data and code deposit (repository and DOI), LLM statement, suggested and excluded reviewers, prior discussion with an Editor.

## Build notes and open technical issues

- **Figs. 3 and 4 are now vector figures** in the journal style, drawn from real data. Fig. 3 uses the July KITTI dump (Kaggle kernel `sallyhaggag/vift-kitti-dump`, in `results_cache/july_dump/kitti_dump`); its curves match the old PNG. Fig. 4 also uses the EuRoC dumps of `pivift_s123_last`, which had not been kept and were regenerated by the new kernel `amrelmadah/pivift-euroc4-s123-infer` (`scripts/kaggle_kernels/kernel_euroc4_s123_infer/`, outputs in `results_cache/euroc4_s123`). The regenerated run reproduces the recorded network ATE exactly (9.08 / 5.11 / 4.06 / 3.63 m, mean 5.47 m). Panel b and the KITTI curve of panel a match the old PNG; the EuRoC curve matches from 2 px upwards, but the new figure has three extra EuRoC bins below 2 px of parallax (about 0.8, 1.1 and 1.5 px, values about 45-75 degrees) that the old PNG did not show. The cause was not found (same binning code and same cached geometry); the caption statement (measuring better by 50-110 degrees on EuRoC) is unchanged in substance. The old PNGs are still in `figures/` for comparison.
- Fig. 3 caption (from the IEEE version): the gyroscope stays within about 0.5 degrees except for the step on drive 05 at 1.13 km, which is in an outage of the inertial unit and appears in both curves; in the plot the gyroscope curve on drive 05 sits near -5 degrees after that step.
- **LaTeX packages installed in the user tree** (`~/Library/texmf`, from the TeX Live 2025 historic repository, no sudo): `cuted` (inside `sttools`), `threeparttable`, `vruler`, `breakurl`, `wrapfig`, `appendix`, `helvetic`, `urw-base35`.
- **`sn-nature.bst` has a local one-function fix** (`format.in.ed.booktitle`): the official December 2024 version makes BibTeX fail with "can't pop an empty literal stack" on every `@inproceedings` entry. The fixed copy is `paper/scireports/sn-nature.bst`; the original is in `template/extracted/`. If Springer Nature's system runs BibTeX itself with its own `.bst`, include the generated `main.bbl` instead.
- `\vbadness=10000` in the preamble hides one harmless warning of `sn-jnl.cls` (an underfull page-footer box on every page, which also appears with no content at all). `\hbadness=4000` around the bibliography hides one mild underfull line in a reference.
- Figure/Fig. 1 and 2 PDFs are built from `figsrc/*.tex` (Nimbus Sans = Helvetica clone, 7 pt, lines 1 pt).
- Not checked: the rendering of `main.pdf` against the journal's own preview, the colour of the corresponding-author line (set by the class), and the Nature-style look of the reference list beyond what `sn-nature.bst` produces.
