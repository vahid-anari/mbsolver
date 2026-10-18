#!/bin/sh
# Build the model documents docs/two_level.pdf and docs/three_level.pdf.
# Needs a LaTeX installation with latexmk. Run from anywhere: sh tools/build_docs.sh
set -e
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root/docs"

version=$(sed -n 's/^version = "\(.*\)"/\1/p' "$root/pyproject.toml")
printf '\\newcommand{\\AppVersion}{%s}\n' "$version" > version.tex

for doc in two_level three_level; do
    latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build "$doc.tex"
    cp "build/$doc.pdf" "$doc.pdf"
done
echo "Built version $version: docs/two_level.pdf, docs/three_level.pdf"
