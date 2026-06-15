# Macro-Financial Relationships — Team Dashboard

An interactive [Streamlit](https://streamlit.io) dashboard that presents the
econometric analysis of seven macro-financial relationships, one per tab, with
the three headline deliverables highlighted on every tab:

| Deliverable | What you see |
|-------------|--------------|
| **① Coefficient** | Headline contemporaneous correlation (on the ADF-correct transform), 95% significance check, theory-peak r, sample size. |
| **② Lead-lag** | Peak lag + leading variable, and an interactive cross-correlation chart (±95% band, auto-peak vs theory-peak markers, divergence callout). |
| **③ Chart** | Interactive scatter at any selectable lag with a live OLS fit, plus a rolling-correlation line with full-sample reference and structural-break markers. |

Plus per-tab **diagnostics**: cointegration (only where statistically valid),
regime sign-flips, the raw-vs-partial common-driver control, detected break
dates, and the full metric table.

## Architecture

The Jupyter notebook **`macro_relationships_master.ipynb`** is the single
*compute* layer. It exports a set of CSVs and PNGs; this app is pure
*presentation* — it just loads those artefacts (no Excel or statsmodels needed
on a teammate's machine).

Artefacts consumed:

```
relationship_summary.csv     wide, one row per relationship (all headline numbers)
dashboard_data.csv           long, every metric as one row
ccf_data.csv                 long, the full cross-correlation function per pair
series_data.csv              long, the aligned transformed (x, y) pair per date
rolling_correlations.csv     long, the 12-quarter rolling correlation per date
breaks_data.csv              long, detected structural-break dates
data_asset_summary.csv       the data-asset catalogue
fig_<n>_*.png                publication figures (300 dpi)
output_gap_diagnostic.png    output-gap orientation sanity check
```

If you change the analysis, re-run the notebook to regenerate these files and
the dashboard updates automatically:

```bash
jupyter nbconvert --to notebook --execute --inplace macro_relationships_master.ipynb
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run dashboard_app.py
```

Then open <http://localhost:8501>. Or use the convenience launcher (it creates
its own virtual environment on first run):

```bash
./run_dashboard.sh
```

## Share it with your team

**Option A — run on a shared host / your machine over the LAN**

```bash
./run_dashboard.sh --lan          # or:
streamlit run dashboard_app.py --server.address 0.0.0.0
```

Teammates on the same network open `http://<your-ip>:8501`.

**Option B — Streamlit Community Cloud (free, public or SSO-gated)**

1. Push this folder (including the exported CSVs/PNGs) to a GitHub repo.
2. At <https://share.streamlit.io> create a new app pointing at
   `dashboard_app.py`.
3. Streamlit installs `requirements.txt` and serves a shareable URL.

> The app reads its data from the folder it lives in, so keep the exported
> CSVs/PNGs alongside `dashboard_app.py` when you deploy.

## Methodology (the four supervising principles)

1. **Non-stationarity** — every series is ADF-tested and routed to the correct
   transform; a levels correlation is never the headline, and cointegration is
   reported only when both series are I(1) in levels.
2. **Lead/lag** — cross-correlation over ±12 quarters; we report both the
   *auto* peak (largest |r|) and the *theory* peak (largest |r| consistent with
   the expected sign and leading direction), and flag where they diverge.
3. **Regime-dependence** — structural breaks (`ruptures`) plus a 12-quarter
   rolling correlation expose regime shifts and sign flips.
4. **Common-driver** — a partial correlation controlling a shared shock
   (energy / policy rate) shows whether each link survives or collapses.
