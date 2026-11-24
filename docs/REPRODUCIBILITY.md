# Reproducibility and execution protocol

## Levels of verification

1. **Packaging verification (completed for the uploaded copy)**: source files inspected; input SHA-256 matched the recorded manifest; the original 3 synthetic integration tests passed in the packaging environment (Python 3.13.5, installed CPU PyTorch 2.10.0).
2. **Recorded result verification (completed)**: the supplied evidence CSVs/manifest reproduce the numbers in the README, and can generate the graph. This does not re-train or validate the scientific results.
3. **Real-data model rerun (NOT completed in the packaging environment)**: user must run the training/evaluation commands below on an appropriately provisioned machine; compare full output tables and environment information, then preserve logs.
4. **Independent replication (NOT completed)**: a separate operator/environment should execute data extraction and training and document any discrepancies.

## Environment

Use a fresh Python 3.11–3.13 environment; the original user-run experiment manifest records **Python 3.13.5 / macOS arm64 / PyTorch 2.15.0 / NumPy 2.5.5**. `requirements.txt` has ranges rather than an exact platform lock. For strict numeric reproducibility, capture `python --version`, `pip freeze`, OS, CPU/GPU/MPS device and package wheel hashes. Bit-exact neural outputs are not guaranteed across hardware/backends.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest -q
python scripts/verify_release.py
python -m pip freeze > results_environment_pip_freeze.txt
```

`results_environment_pip_freeze.txt` is a local reproduction log, not evidence the historical environment has been reconstructed.

## Real data and provenance

Use the exact previously acquired input aggregates for best match. The raw aggregate CSVs are **not committed to GitHub**. Place files in `data/` and check their full SHA-256 digests:

```bash
mkdir -p data
python scripts/verify_release.py --past data/nyc_daily_2024_2025.csv \
  --future data/nyc_daily_2026_jan_sep.csv
```

Fresh live extraction (not guaranteed to reproduce identical bytes: government data can be revised, fields/categories can change):

```bash
python -m src.download --start 2024-01-01 --end 2026-01-01 \
  --output data/nyc_daily_2024_2025.csv
python -m src.download --start 2026-01-01 --end 2026-10-01 \
  --output data/nyc_daily_2026_jan_sep.csv
python scripts/verify_release.py --past data/nyc_daily_2024_2025.csv \
  --future data/nyc_daily_2026_jan_sep.csv
```

If hash verification fails on fresh downloads, **do not silently substitute data** and describe it as exact replication. Investigate schema, source revision, cache, category values and query changes, and report a new run/manifest separately.

## Exact known real-data experiment entrypoints

From the repository root, after installing dependencies and validating data:

```bash
python fair_ablation.py --csv data/nyc_daily_2024_2025.csv \
  --output results/fair_ablation_validation --seeds 7 17 27 37 42

python reviewer_experiments.py --past data/nyc_daily_2024_2025.csv \
  --future data/nyc_daily_2026_jan_sep.csv \
  --output results/reviewer_controls --seeds 7 17 27 37 42 \
  --shuffle-seed 20261124 --bootstrap 300 --block-days 28
```

Add `--cpu` to either command to force CPU execution. The historical `reviewer_experiments.py` reproduces the original (invalid) GBM layout as well as the historical controls; use the corrected extension below for scientific GBM comparisons. These are **not quick smoke tests**: `reviewer_experiments.py` fits 15 neural models plus controls and performs bootstrap calculations, and the results may vary slightly with package/backend versions.

The original frozen 2026 spatial-only and post-hoc subset analyses can also be run separately:

```bash
python frozen_2026_eval.py --past data/nyc_daily_2024_2025.csv \
  --future data/nyc_daily_2026_jan_sep.csv --output results/strict_2026_holdout
python jan_jul_sensitivity.py --past data/nyc_daily_2024_2025.csv \
  --future data/nyc_daily_2026_jan_sep.csv --output results/2026_jan_jul_sensitivity
python audit_2026_coverage.py --historical data/nyc_daily_2024_2025.csv \
  --holdout data/nyc_daily_2026_jan_sep.csv --output results/2026_data_audit
```

Data selection and thresholds are trained on earlier data, and checkpoint/score threshold selection uses 2025 validation only. 2026 outcomes must not be used to fit, tune or recalibrate when claiming a frozen holdout comparison. The inspected 2025 original test and repeatedly explored 2026 subsets are **development history**, not fresh confirmations.

## Latest extension and official audit

```bash
python -u benchmark_extension.py --past data/nyc_daily_2024_2025.csv --future data/nyc_daily_2026_jan_sep.csv --mode gbm --output results/benchmark_extension
python -u benchmark_extension.py --past data/nyc_daily_2024_2025.csv --future data/nyc_daily_2026_jan_sep.csv --mode permutations --permutations 20 --seeds 7 17 27 37 42 --output results/benchmark_extension
python diagnose_nyc_311_category_change.py --output results/category_change_check
```

The permutation extension has 105 model fits, resumes completed runs, and is exploratory. Use fresh output directories for changed configuration/environment. The live API audit may differ from the archived audit.

## Verify existing evidence (no training)

```bash
python scripts/verify_latest_evidence.py
python scripts/plot_reported_results.py
cd paper
python make_figures.py
pdflatex CivicPulse_STGNN.tex
pdflatex CivicPulse_STGNN.tex
cd ..
```

The PDF build needs TeX Live with IEEEtran. `verify_latest_evidence.py` recomputes metrics from archived predictions and checks topology results and corrected GBM; it does not fit models. `paper/analyze_saved_predictions.py` regenerates diagnostics. The archived ZIPs are deliberately tracked as evidence and retained verbatim.

## Dashboard rebuild

```bash
python build_showcase.py --metrics evidence/dashboard_original/metrics.json --communities evidence/dashboard_original/communities.json --csv data/nyc_daily_2024_2025.csv --predictions evidence/dashboard_original/test_predictions.csv --output dashboard/index.html
```

Open locally. This is the original 2025 single-seed graph explorer, not the 2026 evaluation. The built HTML contains derived selected-node daily aggregates. Exact raw input snapshots remain excluded; full independent real-data retraining and platform-specific lock remain pending.
