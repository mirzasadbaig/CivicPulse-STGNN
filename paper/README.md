# CivicPulse-STGNN — revised research discussion draft (v8)

The PDF is a new IEEEtran conference-format manuscript, not a submission or accepted publication. The previous report is superseded; original evidence is retained unchanged. Full real-data training was performed by the author on a Mac, not independently rerun in preparing this manuscript. Saved-output metric checks were performed; see `evidence/verification.json`.

## Rebuild

Run `python make_figures.py` with numpy, pandas and matplotlib installed, then `pdflatex CivicPulse_STGNN.tex` twice (TeX Live with IEEEtran). The figures are supplied as vector PDFs. `analyze_saved_predictions.py` can regenerate the saved-prediction diagnostics using the two original result archives:

```sh
python analyze_saved_predictions.py evidence/original_archives/CivicPulse_Permutation_Results.zip evidence/original_archives/CivicPulse_Reviewer_Experiment_Results.zip regenerated_diagnostics
```

`../benchmark_extension.py` is the experiment extension source. Run it from the repository root with the two verified local input CSVs. This manuscript package is not a complete standalone training repository. Use the original manifests for configuration; the source CSV SHA-256 digests are recorded in `evidence/historical_manifest.json`.

## Changes through this version

- Replaces invalid GBM scores with the corrected node-history layout and validation-only search. The change combines a bug fix and a search, so the improvement cannot be attributed solely to layout.
- Reports the initial shuffled control plus 20 additional topologies (105 extension runs), with topology-level mean and SD. Ranks and tail fractions are descriptive because 2026 was previously inspected.
- Preserves persistence as the strongest tested predictor on both reported windows; the neural model has modest AP gains over no-mixing and sampled relabelings.
- Clarifies that full-node permutations disrupt category-aware alignment as well as spatial alignment; they do not isolate geographic proximity.
- Includes monthly threshold diagnostics (retrospective oracle F1 is NOT a valid test score), year-over-year counts, and exact source-label discontinuity. The same-month comparison is descriptive and includes dates used for training/validation in 2025.
- Removes reviewer-oriented labels, moves community details to a supplement, fixes architecture diagram and includes seed SD in the main results table.
- Retains an explicit AI-assistance disclosure and no claim of independent training replication or publication.

Remaining: author/mentor review, public repo release and rights/license check, future untouched evaluation, more constrained nulls and persistence-aware/calibrated models. See the manuscript limitations and `REFERENCE_CHECK.md`.

Version 8 retains the supervisor and internship credit introduced in version 7, archives the original five category-threshold runs, and verifies all 15 metric rows and 25 cutoffs. Table IV now reports sample SD. Category-preserving graph permutations remain unrun. From repository root, run `python scripts/verify_category_thresholds.py` to check the evidence.
