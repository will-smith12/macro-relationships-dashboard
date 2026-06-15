# Macro-Financial Relationships — US & Canada

An interactive econometric analysis of eight macro-financial relationships,
each studied under four supervising principles and reported **side-by-side for
the United States and Canada**. The headline deliverable is a shareable
[Streamlit](https://streamlit.io) dashboard backed by a reproducible Jupyter +
R analysis pipeline.

<!-- After deploying on Streamlit Community Cloud, paste the live URL here: -->
**▶️ Live demo:** <https://macro-relationships-dashboard-vfbp6xq4ygvxnapt77pm6j.streamlit.app>

---

## The relationships

| # | Relationship | Expectation |
|---|--------------|-------------|
| 1 | Okun's Law — ΔUnemployment vs GDP growth | inverse (+ a time-varying TVP/Kalman slope) |
| 2 | Phillips Curve — Unemployment vs CPI inflation | inverse, unstable / regime-dependent |
| 3 | Output gap vs CPI inflation | positive, inflation lagged |
| 4 | CPI inflation vs short-term policy rate | Taylor-rule positive |
| 5 | Yield-curve slope (10Y-3M) vs GDP growth | slope leads growth |
| 6 | **Energy prices vs CPI inflation** | **threshold VECM** — long-run pass-through, US & Canada |
| 7 | Interest-rate differential vs exchange rate | higher relative rates → appreciation |
| 8 | VIX vs GDP growth | uncertainty leads activity (negative) |

### Highlighted analyses
- **Energy → CPI (Threshold VECM):** a full Hansen–Seo threshold cointegration
  pipeline. Both series are I(1) and cointegrated; the linear-cointegration null
  is rejected in favour of a two-regime model, validated out-of-sample against an
  AR(2) benchmark — reported for the US and Canada together.
  (US long-run elasticity ≈ 1.10, Canada ≈ 0.66.)
- **Time-varying Okun slope:** a Beaton-style random-walk (TVP) Okun coefficient
  estimated with a Kalman smoother and 95% band, exposing regime shifts a single
  static coefficient would miss.

## The four supervising principles

1. **Non-stationarity** — every series is ADF-tested and routed to the correct
   transform; a levels correlation is never the headline, and cointegration is
   reported only when both series are I(1).
2. **Lead/lag** — cross-correlation over ±12 quarters; both the auto peak and the
   theory peak are reported, with divergences flagged.
3. **Regime-dependence** — structural breaks (`ruptures`) plus a rolling
   correlation expose regime shifts and sign flips.
4. **Common-driver** — a partial correlation controlling a shared shock (energy /
   policy rate) shows whether each link survives or collapses.

## Architecture

The notebook **`macro_relationships_master.ipynb`** is the single *compute* layer.
It exports the CSVs and PNGs in this folder; **`dashboard_app.py`** is pure
*presentation* and just loads those artefacts — no Excel or statsmodels needed on
a viewer's machine. The threshold-VECM estimation lives in the R scripts
(`estimate_tvecm.R`, `estimate_tvecm_canada.R`) with out-of-sample validation in
`vecm_forecast.py`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run dashboard_app.py
```

Then open the URL Streamlit prints (default <http://localhost:8501>).

## Deploy on Streamlit Community Cloud

1. Push this repo to GitHub (the exported CSVs/PNGs are included, so the app is
   self-contained).
2. At <https://share.streamlit.io>, create a new app pointing at
   `dashboard_app.py` on the `main` branch.
3. Streamlit installs `requirements.txt` and serves a shareable URL — paste it at
   the top of this README.

## Reproducing the analysis

The raw master spreadsheet is **not** included (it is a proprietary source
workbook); the derived series live in the committed CSVs (`fred_data*.csv`,
`relationship_summary.csv`, etc.), which is all the dashboard needs. To refresh
US series from FRED, set a key first:

```bash
export FRED_API_KEY='your_key'   # https://fred.stlouisfed.org/docs/api/api_key.html
python fetch_fred_data.py
```

See **`README_dashboard.md`** for the dashboard internals and **`README_mcp.md`**
for the companion MCP server.

## Repository layout

```
dashboard_app.py              the Streamlit dashboard (presentation layer)
macro_relationships_master.ipynb   the compute notebook (exports CSVs/PNGs)
estimate_tvecm.R / _canada.R  Hansen–Seo threshold VECM (US / Canada)
vecm_forecast.py              out-of-sample VECM vs AR(2) validation
okun_tvp.py                   time-varying (TVP/Kalman) Okun slope
*.csv                         exported analysis artefacts (dashboard inputs)
*.png                         publication figures (300 dpi)
requirements.txt              dashboard dependencies (streamlit/pandas/numpy/plotly)
```
