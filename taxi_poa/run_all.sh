#!/usr/bin/env bash
# Reproduces every result, figure, table and the report from scratch (fixed seeds; ~3-5 minutes on 2 cores).
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH=.
mkdir -p results figures

echo "[1/8] tests";                python -m pytest -q tests | tee /dev/stderr | tail -1 > results/test_summary.txt
echo "[2/8] main sweeps (V B D C A)"; python -m game.sweep V B D C A --seeds 100 --jobs 2
echo "[3/8] equity + convergence sweep (E)"; python -m game.sweep_equity --seeds 100 --jobs 2
echo "[4/8] learning sweep (L)";   python -m game.sweep_learning --seeds 40 --T 300 --jobs 2
echo "[5/8] verification scripts"; python audit/opt_is_matching.py; python audit/mutation_check.py; python audit/pilot_poa_explosion.py; python audit/long_horizon.py
echo "[6/8] analysis tables";      python audit/analyze.py > results/analysis_output.txt
echo "[7/8] figures";              python audit/make_figures.py
echo "[8/8] report";               python report/make_tables.py
if command -v pdflatex >/dev/null; then (cd report && pdflatex -interaction=nonstopmode main.tex >/dev/null && pdflatex -interaction=nonstopmode main.tex >/dev/null && echo "report/main.pdf built"); else echo "pdflatex not found: skipped PDF"; fi
