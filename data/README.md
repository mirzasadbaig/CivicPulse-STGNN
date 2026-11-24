# Local data directory (ignored by Git)

Place the exact CSVs here to match the recorded experiment:

- `nyc_daily_2024_2025.csv` — SHA-256 `5f98e176da8168a4b5dbc409c958b5c58322d835d43ad8f96cd289110d012fec`
- `nyc_daily_2026_jan_sep.csv` — SHA-256 `e442a3de8c0dd7652418816a28b0c118220b8921be910521cba163e8c25f8afe`

Raw grouped city data are not included in this release candidate. Use `python scripts/verify_release.py --past ... --future ...` to check exact input identity. Government source updates may change results of a fresh download.
