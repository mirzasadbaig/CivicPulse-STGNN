# Table IV: verified original experiment

The original Mac result archive is included at `paper/evidence/original_archives/CivicPulse_Category_Threshold_Results.zip`. It contains the manifest, labels, all five seed JSON/NPZ pairs, per-run CSV and topology means.

From repository root:

```sh
python scripts/verify_category_thresholds.py
```

The verifier checks the two dataset digests against the previous manifest, 2026 labels/dates against archived GBM labels, and reselects 25 cutoffs using only the validation scores and labels. All 15 metric rows match. The 2026 spatial prediction arrays are identical to the previous extension. Recomputed per-seed CSV, cutoff CSV and mean/sample-SD CSV are in `paper/evidence/final_review/`. This is independent metric recalculation, not independent training. No category-preserving topology experiment is claimed.
