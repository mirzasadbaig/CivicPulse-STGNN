# Seven-minute technical demonstration

Open the [paper](../paper/CivicPulse_STGNN.pdf), [benchmark figure](../assets/benchmark_ap.png), [topology figure](../assets/permutation.png) and `dashboard/index.html` in advance.

- 0:00–0:45 — Research question: can graph-aware recurrence beat temporal persistence on elevated complaint demand? This is not crisis prediction.
- 0:45–1:40 — NYC 311, 20 ZIP codes × four categories, 80 graph nodes; within-ZIP and same-category distance-weighted links. Explain 28-day observed history and next-48-hour target relative to the training upper quartile.
- 1:40–2:30 — Chronological training/validation and frozen 2026 evaluation, matched features, five seeds, and why the already inspected windows are exploratory.
- 2:30–3:30 — Persistence wins (.583 F1/.613 AP versus spatial .479/.604 in Jan–Jul). Explain the GBM tensor layout bug and corrected .597 AP, with 2025-only search.
- 3:30–4:30 — 20 additional full-node relabelings: geographic graph ranks first in AP, third in F1; category alignment is disrupted too. Explain what this does and does not establish.
- 4:30–5:30 — April F1 .034, AP .310, retrospective best-threshold F1 .402 (not a test result); category-specific thresholds produce F1 .484/.439 in the two later windows (verified five-seed means); year-over-year count changes and missing old labels in Aug–Sep.
- 5:30–6:30 — Dashboard: choose observed demand, original saved forecast and December community views; 731/108/21 date coverage. Show code and evidence manifest. The dashboard shows the earlier 2025 run, not 2026 results.
- 6:30–7:00 — Ask for feedback on persistence-aware models, topology-specific controls and a new untouched evaluation. The contribution is a benchmark and failure-mode analysis, not a new GNN architecture.

Practice explaining why AP differs from F1, why the tensor transpose matters, and why the new results do not establish causality. State that full real-data fitting has not been independently replicated.
