#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/paper"
PDFLATEX="${PDFLATEX:-$HOME/Library/TinyTeX/bin/universal-darwin/pdflatex}"
if [ ! -x "$PDFLATEX" ]; then
  PDFLATEX="$(command -v pdflatex)"
fi

"$PDFLATEX" -interaction=nonstopmode -halt-on-error symbolic_lbbnn_paper.tex
"$PDFLATEX" -interaction=nonstopmode -halt-on-error symbolic_lbbnn_paper.tex

echo "[OK] Compiled paper/symbolic_lbbnn_paper.pdf"
grep -iE "warning.*undefined|error|overfull" symbolic_lbbnn_paper.log | head -n 30 || true
