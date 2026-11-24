# Data provenance and frozen population

**Canonical source:** NYC Open Data 311 Service Requests from 2020 to Present, dataset `erm2-nwe9`:
https://data.cityofnewyork.us/d/erm2-nwe9

The repository extractor uses a public Socrata API query grouped by calendar day of `created_date`, `incident_zip`, and **exact** `complaint_type` field value, with `count(*)`, mean latitude and mean longitude. Monthly cached requests can yield a different extraction if city records or field meanings have changed. Optional `SOCRATA_APP_TOKEN` is read from the environment and must never be committed.

**Recorded inputs (full SHA-256 digests verified against supplied ZIP):**

```text
2024-01-01 through 2025-12-31 (end exclusive 2026-01-01)
data/nyc_daily_2024_2025.csv
5f98e176da8168a4b5dbc409c958b5c58322d835d43ad8f96cd289110d012fec

2026-01-01 through 2026-09-30 (end exclusive 2026-10-01)
data/nyc_daily_2026_jan_sep.csv
e442a3de8c0dd7652418816a28b0c118220b8921be910521cba163e8c25f8afe
```

Raw source aggregate files are deliberately **not committed**; each owner must obtain, preserve, and verify the permitted data from the source. The two source files supplied during packaging produced the full hashes above. These are precise input identities, not proof that the government source has not changed afterward.

**Frozen population:** select top 20 ZIP codes using training-period complaint activity; cross with four exact historical complaint categories `HEAT/HOT WATER`, `Water System`, `Street Condition`, `Sewer`, giving 80 study nodes. They are neither all NYC ZIP codes nor all 311 problem types. Training-only centroids and count/scaler/label cutoffs are frozen for the 2026 scoring protocol.

**Known source discontinuity:** in the original 2026 export the exact `Water System` and `Sewer` labels are absent in August and September. `Water Maintenance` and `Sewer Maintenance` exist but also coexisted with the old labels in July. An official one-to-one crosswalk has **not** been established. Do not relabel without independent official evidence and a new, clearly labeled analysis. See `evidence/source_audit/` for the user-run audit files. Official city dataset/category nomenclature may change; check https://www.nyc.gov/opendata/news/all-news/311-Service-Requests-Updates before new extractions.

**Data licensing:** public availability does not waive source attribution, correct representation, or review of NYC Open Data's terms. Confirm appropriate rights and institutional restrictions before including raw extracts, dashboards, screenshots or derived plots in a public repository.
