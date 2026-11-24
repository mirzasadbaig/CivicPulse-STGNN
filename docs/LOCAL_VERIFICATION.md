# Release packaging checks

Checks performed against the assembled release in the packaging environment:

- Four pytest tests passed (three original integration tests plus node-history layout regression).
- Both archived original aggregate CSVs matched the SHA-256 digests in the experiment manifest; the raw aggregate files are not included.
- Recomputed 210 period-level rows from 105 archived prediction arrays, verified 20 unique permutations, and compared the topology and monthly diagnostic tables to the supplied evidence.
- Recomputed 2026 corrected GBM AP, F1, precision and recall from saved predictions.
- Rebuilt dashboard from original inputs; 80 nodes, 731 observation dates, 276 edges, 108 saved forecast dates, 21 community snapshots. Headless Chromium loaded the HTML content, advanced from Jan 1 to Jan 2, and jumped to Dec 21 (six communities) without JavaScript errors. The container blocked file:// navigation, so local file opening on the author's Mac remains a manual check.
- Local Git staging check confirmed the original three evidence ZIPs are included, with no raw input CSVs, checkpoints, virtual environments or Python caches staged.

These checks are not independent real-data model fitting, a peer review, a claim of operational forecasting validity, or a guarantee that every browser interaction works. The original training environment is recorded in the manifests; dependencies are version ranges rather than a bit-exact lock. CI is configured but has not yet run on GitHub.

Version 8 adds the fourth original archive and independently checks all 15 category-threshold metric rows and 25 cutoffs.
