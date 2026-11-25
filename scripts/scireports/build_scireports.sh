#!/bin/bash
# Build the Scientific Reports version (run only after the go-ahead).
#   1. TikZ figures  paper/scireports/figsrc/fig_*.tex  ->  paper/scireports/figures/*.pdf
#   2. main.pdf (pdflatex, bibtex, pdflatex x2) and supplementary.pdf
#   3. error / warning check on the logs
# The two matplotlib figures (heading_error.pdf, measure_vs_learn.pdf) are made separately,
# see scripts/scireports/RUN_FIGURES.md.
set -u
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SR="$ROOT/paper/scireports"
TMP="${TMPDIR:-/tmp}/scireports_build"; mkdir -p "$TMP/figs"
cd "$SR/figsrc" || exit 1
for pair in fig_timing:timing fig_pipeline:pipeline fig_learned:learned_branch fig_pivift:pivift fig_gate:gate_decisions; do
  src="${pair%%:*}"; out="${pair##*:}"
  pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$TMP/figs" "$src.tex" >"$TMP/figs/$src.out" 2>&1 \
    || { echo "FIGURE FAILED: $src (see $TMP/figs/$src.log)"; continue; }
  cp "$TMP/figs/$src.pdf" "$SR/figures/$out.pdf"
done
cd "$SR" || exit 1
pdflatex -interaction=nonstopmode -file-line-error main.tex >/dev/null 2>&1
bibtex main >/dev/null 2>&1
pdflatex -interaction=nonstopmode -file-line-error main.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode -file-line-error main.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode -file-line-error supplementary.tex >/dev/null 2>&1
pdflatex -interaction=nonstopmode -file-line-error supplementary.tex >/dev/null 2>&1
echo "--- errors (lines starting with '!' or file:line:) ---"
grep -nE '^!|^\./.*:[0-9]+:' main.log supplementary.log | head -40
echo "--- warnings ---"
grep -nE 'Overfull|Underfull|undefined|Warning' main.log supplementary.log main.blg | head -60
echo "--- pages ---"
for f in main supplementary; do printf "%s: " $f; pdfinfo $f.pdf 2>/dev/null | grep Pages || echo "no pdf"; done
