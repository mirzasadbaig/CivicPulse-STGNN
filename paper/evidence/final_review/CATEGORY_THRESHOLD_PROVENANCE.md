# Category-threshold experiment: archived and verified

Original user-supplied archive: `../original_archives/CivicPulse_Category_Threshold_Results.zip`. Five seeds (7, 17, 27, 37, 42), spatial topology only. Validation and later-period label/score arrays, cutoff JSONs and original summaries are preserved. The historical console transcription is retained for audit, but superseded by `category_threshold_verified_runs.csv` and `category_threshold_verified_summary.csv`.

Run `python scripts/verify_category_thresholds.py` from repository root. All 15 metric rows and 25 cutoffs match independent recalculation; 2026 prediction arrays match the prior spatial extension exactly. No independent model fitting was performed. `category_f1` is pooled F1 after applying four category-specific cutoffs, not macro-F1. AP is unchanged for the same raw scores. Category-preserving graph permutations remain unrun.
