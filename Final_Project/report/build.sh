#!/usr/bin/env bash
# Rebuilds both PDFs (main academic report + short LinkedIn showcase) from
# the LaTeX sources in this directory, refreshing figures/ from
# ../saved_images/ first.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

cp ../saved_images/*.png figures/

for doc in main linkedin; do
  pdflatex -interaction=nonstopmode "$doc.tex" >/dev/null
  pdflatex -interaction=nonstopmode "$doc.tex" >/dev/null   # 2nd pass: refs/labels/TOC
  echo "Built $doc.pdf"
done
