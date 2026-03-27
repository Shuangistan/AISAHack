#!/usr/bin/env bash
# Build the LaTeX document to PDF.
# Usage: ./build.sh

set -euo pipefail
cd "$(dirname "$0")"

pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex   # second pass for TOC / refs

echo ""
echo "Done. Output: $(pwd)/main.pdf"
