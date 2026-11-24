# Final review action record (2026-11-24)

Completed: naive validation metrics (same 108 anchors); Brier scores and reliability bins from archived predictions; manuscript edits; reference 9 and 10 publisher verification; reproducibility appendix; deletion of dangling Louvain reference from the main paper. Community visualizations remain separate dashboard material.

New diagnostics: `paper/evidence/final_review/`. Both exact aggregate hashes and the reconstructed 2026 labels/dates were verified before calculation. The Brier score is probability accuracy, not a pure calibration measure. Raw margins from naive rules are not probabilities.

Author review and host permission are not completed by these software checks. The repository is now available at https://github.com/mirzasadbaig/CivicPulse-STGNN. Its current public visibility is not evidence of release permission. Do not publish or externally distribute internship-derived material without applicable clearance. Keep the author approval checklist outside the manuscript, without misrepresenting the manuscript as accepted or cleared. Confirm the host's AI rules and authorship, and independently inspect code, numerical tables, references and figures. IEEE requires disclosure of generated text, figures and code; see https://open.ieee.org/author-guidelines-for-artificial-intelligence-ai-generated-text/ and https://ieee-cas.org/publications/using-ai-generated-content-ieee-article. JMI's IPR and academic-integrity framework is at https://jmi.ac.in/upload/menuupload/research_ipr_policy.pdf and https://jmi.ac.in/RESEARCH-%26-INNOVATION/University-Academic-Integrity-Panel/Notifications. Neither establishes approval for this internship.

Reference 9: https://www.nature.com/articles/s41598-026-49982-2 (Yang, Lu, Huang; NYC 311, differing tasks). Reference 10: https://www.sciencedirect.com/science/article/pii/S1574119220301449 (Raj, Ramesh, Seetharam, DeFazio; response-time regression). Neither is a directly comparable binary demand-quantile benchmark.

## Repeat diagnostics
From repository root, with original CSVs locally available:
```
python validation_calibration.py --past data/nyc_daily_2024_2025.csv --future data/nyc_daily_2026_jan_sep.csv
```

## Category-threshold extension (verified saved predictions) and optional topology control (unrun)
Joint ZIP permutation retains categories and co-location edges by applying the same ZIP relabeling to all four categories. It does not preserve each individual node's degree. The generator was checked on the exact dataset for 20 unique permutations, category preservation and edge-weight multiset preservation. A spatial-only run has now been verified; the category-preserving permutation training is still unexecuted. Each model uses the original protocol and selects one cutoff per category on 2025 validation F1 only. Comparing pooled F1 at these cutoffs versus global cutoff is exploratory; AP is unchanged for identical raw scores. The spatial-only `--permutations 0` run is archived and independently verified at metric level; use `--permutations 20` only if pursuing the optional 105-fit topology control. Use separate output directories for different configurations. All selected cutoffs and predictions are saved. Do not pool these new refits with historical runs or claim new untouched test results.
```
python category_control.py --past data/nyc_daily_2024_2025.csv --future data/nyc_daily_2026_jan_sep.csv --check-only
python -u category_control.py --past data/nyc_daily_2024_2025.csv --future data/nyc_daily_2026_jan_sep.csv --permutations 0 --output results/category_thresholds
python -u category_control.py --past data/nyc_daily_2024_2025.csv --future data/nyc_daily_2026_jan_sep.csv --permutations 20 --output results/category_control
```

Verified category-cutoff F1: 0.525846 (validation), 0.483566 (Jan–Jul), 0.438665 (Jan–Sep). See `paper/evidence/final_review/CATEGORY_THRESHOLD_PROVENANCE.md`. All five per-seed artifacts are archived; run `python scripts/verify_category_thresholds.py`.
