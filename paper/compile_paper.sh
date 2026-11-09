#!/usr/bin/env bash
set -euo pipefail
export PATH="/Users/aliaksandrhome/Library/TinyTeX/bin/universal-darwin:/Library/TeX/texbin:/opt/homebrew/bin:/usr/local/bin:$PATH"
cd "$(dirname "$0")"

if command -v pdflatex &> /dev/null; then
    pdflatex -interaction=nonstopmode symbolic_lbbnn_paper.tex
    if grep -q "\\\\bibdata" symbolic_lbbnn_paper.aux 2>/dev/null; then
        bibtex symbolic_lbbnn_paper
    fi
    pdflatex -interaction=nonstopmode symbolic_lbbnn_paper.tex
    pdflatex -interaction=nonstopmode symbolic_lbbnn_paper.tex
elif command -v tectonic &> /dev/null; then
    tectonic symbolic_lbbnn_paper.tex
fi
echo "Successfully compiled $(pwd)/symbolic_lbbnn_paper.pdf"
