#!/bin/bash
# Build a clean folder that Overleaf compiles (pdfLaTeX + BibTeX) with one standalone main.tex,
# and check it: a relocated copy must compile on its own and its text must equal the text of
# the manuscript built from the split sources.
#   bash scientific_reports_paper/make_overleaf_folder.sh
# Writes scientific_reports_paper/overleaf_scireports_<date>/: main.tex (flattened by
# flatten_main.py: Methods, tables and number values inlined), refs.bib with the compiled
# main.bbl, the Springer Nature class and BibTeX style, and the six figure PDFs. Upload the
# folder's contents to a new Overleaf project; main.tex is the root document.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
PY="${PY:-$REPO/.venv/bin/python}"
TECTONIC="${TECTONIC:-$HOME/.local/bin/tectonic}"
OUT="$HERE/overleaf_scireports_$(date +%Y_%m_%d)"
FILES=(refs.bib sn-jnl.cls sn-nature.bst
       figures/fig1_teaser.pdf figures/fig2_method.pdf figures/fig3_pose_gap.pdf
       figures/fig4_results.pdf figures/fig5_analysis.pdf figures/fig6_qualitative.pdf)

cd "$HERE"
"$TECTONIC" -X compile --keep-intermediates main.tex >/dev/null  # reference PDF and main.bbl
rm -rf "$OUT"
mkdir -p "$OUT/figures"
"$PY" "$HERE/flatten_main.py" "$OUT/main.tex"
for f in "${FILES[@]}" main.bbl; do cp "$f" "$OUT/$f"; done

echo "non-ASCII characters in main.tex / refs.bib (check each for pdflatex):"
grep -nP "[^\x00-\x7F]" "$OUT/main.tex" "$OUT/refs.bib" || echo "  none"
echo "placeholders left in main.tex:"
grep -n "to be added\|To be completed\|To be confirmed" "$OUT/main.tex" || echo "  none"

CHECK="$(mktemp -d)"
trap 'rm -rf "$CHECK"' EXIT
cp -r "$OUT/." "$CHECK"
if command -v pdflatex >/dev/null && command -v bibtex >/dev/null; then
    (cd "$CHECK" && rm -f main.bbl && pdflatex -interaction=nonstopmode -halt-on-error main.tex \
        >/dev/null && bibtex main >/dev/null && pdflatex -interaction=nonstopmode main.tex \
        >/dev/null && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null)
    ENGINE=pdflatex
else
    (cd "$CHECK" && rm -f main.bbl && "$TECTONIC" -X compile main.tex >/dev/null)
    ENGINE="tectonic (pdflatex is not installed here)"
fi
"$PY" -I - "$HERE/main.pdf" "$CHECK/main.pdf" "$ENGINE" <<'EOF'
import sys

import pymupdf

ref, new, engine = sys.argv[1:4]
text = [" ".join(page.get_text() for page in pymupdf.open(p)).split() for p in (ref, new)]
print(f"relocated copy compiles with {engine}: {pymupdf.open(new).page_count} pages")
print("text identical to the split-source build:", text[0] == text[1])
if text[0] != text[1]:
    sys.exit("flattened manuscript differs from the split-source build")
EOF
(cd "$OUT" && find . -type f | sort | xargs du -h)
echo "wrote $OUT"
