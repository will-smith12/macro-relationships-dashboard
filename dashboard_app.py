"""
Macro-Financial Relationships — interactive team dashboard
==========================================================

A shareable Streamlit front-end for the econometric analysis produced by
``macro_relationships_master.ipynb``.  The notebook stays the single COMPUTE
layer; this app is pure PRESENTATION — it loads the exported CSVs / PNGs and
renders one tab per relationship with the three headline deliverables
highlighted:  COEFFICIENT, LEAD-LAG, CHART.

Run locally:
    pip install -r requirements.txt
    streamlit run dashboard_app.py

The four supervising principles behind every number:
  1. Non-stationarity  — ADF-routed transforms; cointegration only when valid.
  2. Lead/lag          — cross-correlation over ±12 quarters (auto vs theory peak).
  3. Regime-dependence — structural breaks + rolling correlation (sign flips).
  4. Common-driver      — raw vs partial correlation controlling a shared shock.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Macro-Financial Relationships",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Plain-English "why this matters" notes, keyed by relationship number.
WHY = {
    1: ("**Okun's Law** ties the *change* in unemployment to real GDP growth: "
        "when the economy grows above trend, firms hire and joblessness falls. "
        "We difference unemployment so both sides are flows — a dimensionally "
        "consistent comparison. Expect a clear **negative** contemporaneous link."),
    2: ("The **Phillips Curve** posits an inverse trade-off between unemployment "
        "and inflation. It is notoriously **unstable / regime-dependent** — the "
        "relationship weakened after the 1990s, so Principle 3 (breaks & rolling "
        "correlation) and Principle 4 (common-driver) bite hardest here."),
    3: ("The **output gap** (actual vs potential GDP) is a textbook driver of "
        "inflation: an economy running hot pushes prices up, typically with a "
        "**lag of a few quarters**. We expect a **positive** correlation with the "
        "**gap leading** inflation — Principle 2 (lead/lag) is decisive."),
    4: ("Under a **Taylor rule**, the central bank raises the policy rate in "
        "response to inflation, so inflation should **lead** the rate with a "
        "**positive** sign. Principle 4 matters: energy shocks move both, so we "
        "check the link survives controlling for energy."),
    5: ("The **yield-curve slope** (10Y-3M) is a classic recession predictor: an "
        "inversion today signals weak growth 6–18 months out. We therefore expect "
        "the **slope to lead** GDP growth at a **positive** lag — a pure Principle 2 "
        "story a static correlation matrix would miss."),
    6: ("**Energy prices** pass through to headline inflation fast and strongly, but the "
        "raw correlation hides the real structure. Both series are **I(1) and "
        "cointegrated**, so the honest model is a **threshold vector error-correction "
        "model (TVECM)**: energy and CPI share a long-run equilibrium and error-correct "
        "back toward it — but **at different speeds in turbulent vs normal regimes** "
        "(Hansen & Seo 2002). This tab reports the long-run elasticity, the regime "
        "structure, and an out-of-sample test against an AR(2) benchmark."),
    7: ("**Interest-rate differentials** drive exchange rates: higher relative "
        "rates attract capital and appreciate the currency. This pair is a **prime "
        "suspect for instability** (Principle 3). The deliverable here is the "
        "**graph**; the coefficient is secondary."),
    8: ("The **VIX** — equity-market implied volatility — is the real-time gauge of "
        "financial-market **uncertainty**. Bloom (2009) shows uncertainty spikes "
        "precede drops in activity, so expect a **negative** link with the **VIX "
        "leading** GDP growth. This is the **first market-based channel** in the set "
        "and is dominated by crisis spikes (2008, 2020) — Principle 3 bites hardest."),
}


# --------------------------------------------------------------------------- #
# Data loading (cached)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_all():
    summary = load_csv("relationship_summary.csv")
    dash = load_csv("dashboard_data.csv")
    ccf = load_csv("ccf_data.csv")
    series = load_csv("series_data.csv")
    rolling = load_csv("rolling_correlations.csv")
    breaks = load_csv("breaks_data.csv")
    catalogue = load_csv("data_asset_summary.csv")
    okun_tvp = load_csv("okun_tvp.csv")
    okun_tvp_reg = load_csv("okun_tvp_regimes.csv")

    for df, col in [(series, "date"), (rolling, "date"), (breaks, "break_date"),
                    (okun_tvp, "date")]:
        if not df.empty and col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return (summary, dash, ccf, series, rolling, breaks, catalogue,
            okun_tvp, okun_tvp_reg)


SUMMARY, DASH, CCF, SERIES, ROLLING, BREAKS, CATALOGUE, OKUN_TVP, OKUN_TVP_REG = load_all()

if SUMMARY.empty:
    st.error(
        "No analysis artefacts found in this folder. Run the notebook first:\n\n"
        "`jupyter nbconvert --to notebook --execute --inplace "
        "macro_relationships_master.ipynb`"
    )
    st.stop()

# Unique relationships (one tab each), ordered by relationship number. The
# summary now carries one row per country, so the Relationship column repeats.
def _rel_sort_key(name):
    return _rel_number_raw(name)


def _rel_number_raw(name: str) -> int:
    try:
        return int(str(name).split("·")[0].strip().split(" ")[0])
    except Exception:
        return 0


REL_NAMES = sorted(SUMMARY["Relationship"].unique().tolist(), key=_rel_sort_key)

# Country display order + code/flag mapping for figure prefixes and labels.
COUNTRY_ORDER = ["US", "Canada", "Cross-border"]
CC_CODE = {"US": "us", "Canada": "ca", "Cross-border": "xb"}
CC_LABEL = {"US": "🇺🇸 United States", "Canada": "🇨🇦 Canada",
            "Cross-border": "🌐 Cross-border (US ↔ Canada)"}


def countries_for(rel_name: str) -> list:
    """Countries present for a relationship, in display order."""
    if "Country" not in SUMMARY.columns:
        return ["US"]
    have = SUMMARY[SUMMARY["Relationship"] == rel_name]["Country"].unique().tolist()
    return [c for c in COUNTRY_ORDER if c in have]


def rel_number(name: str) -> int:
    """Parse the leading integer from a relationship label like '3 · ...'."""
    try:
        return int(str(name).split("·")[0].strip().split(" ")[0])
    except Exception:
        return 0


def short_label(name: str) -> str:
    """A compact tab label, e.g. '3 · Output gap vs CPI'."""
    parts = str(name).split("—")
    head = parts[0].strip()
    tail = parts[-1].strip() if len(parts) > 1 else ""
    return f"{head} {tail}".strip() if tail else head


def fig_path(n: int, cc: str = "us") -> Path | None:
    hits = sorted(DATA_DIR.glob(f"fig_{cc}_{n}_*.png"))
    if hits:
        return hits[0]
    hits = sorted(DATA_DIR.glob(f"fig_{n}_*.png"))  # legacy unprefixed fallback
    return hits[0] if hits else None


def metric_lookup(rel: str, country: str = "US") -> dict:
    """Return {metric_name: value} for one relationship+country from the long table."""
    if DASH.empty:
        return {}
    sub = DASH[DASH["relationship"] == rel]
    if "country" in sub.columns:
        sub = sub[sub["country"] == country]
    return dict(zip(sub["metric_name"], sub["metric_value"]))


def fmt(v, nd=3, dash="—"):
    if v is None:
        return dash
    if isinstance(v, float) and np.isnan(v):
        return dash
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("📈 Macro Relationships")
    st.caption(
        "Interactive companion to *macro_relationships_master.ipynb*. "
        "Each tab analyses one macro-financial relationship under four "
        "supervising econometric principles."
    )
    st.divider()
    st.subheader("Methodology")
    st.markdown(
        "- **P1 Non-stationarity** — ADF-routed transforms; cointegration "
        "only when both series are I(1) in levels.\n"
        "- **P2 Lead/lag** — cross-correlation ±12 quarters; *auto* peak "
        "(largest |r|) vs *theory* peak (sign- & direction-consistent).\n"
        "- **P3 Regime** — structural breaks + 12-quarter rolling correlation; "
        "we flag sign flips.\n"
        "- **P4 Common-driver** — raw vs partial correlation controlling a "
        "shared shock (energy / policy rate)."
    )
    st.divider()
    st.subheader("Download data")
    for fname in [
        "relationship_summary.csv", "dashboard_data.csv", "ccf_data.csv",
        "series_data.csv", "rolling_correlations.csv", "breaks_data.csv",
        "data_asset_summary.csv",
    ]:
        fp = DATA_DIR / fname
        if fp.exists():
            st.download_button(
                f"⬇ {fname}", fp.read_bytes(), file_name=fname,
                mime="text/csv", use_container_width=True, key=f"dl_{fname}",
            )
    st.caption(f"Data folder: `{DATA_DIR}`")


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
tab_labels = ["🏠 Overview"] + [
    "⚡ Energy & Inflation (TVECM)" if rel_number(n) == 6 else short_label(n)
    for n in REL_NAMES
]
tabs = st.tabs(tab_labels)


# ====================  OVERVIEW  =========================================== #
def style_summary(df: pd.DataFrame):
    def colour_flip(v):
        if str(v).strip().lower() == "yes":
            return "background-color:#fde2e1;color:#8a1c12;font-weight:600"
        if str(v).strip().lower() == "no":
            return "background-color:#e3f6e5;color:#1c6b2a"
        return ""

    def colour_survive(v):
        s = str(v).strip().lower()
        if s in ("yes", "survives"):
            return "background-color:#e3f6e5;color:#1c6b2a;font-weight:600"
        if s in ("no", "collapses"):
            return "background-color:#fde2e1;color:#8a1c12;font-weight:600"
        return ""

    def colour_div(v):
        if str(v).strip() not in ("—", "", "nan"):
            return "background-color:#fff4d6;color:#7a5400"
        return ""

    sty = df.style
    if "Sign-flips across regimes?" in df.columns:
        sty = sty.map(colour_flip, subset=["Sign-flips across regimes?"])
    if "Survives common-driver control?" in df.columns:
        sty = sty.map(colour_survive,
                      subset=["Survives common-driver control?"])
    if "Auto/Theory divergence" in df.columns:
        sty = sty.map(colour_div, subset=["Auto/Theory divergence"])
    return sty


with tabs[0]:
    st.title("Macro-Financial Relationships — Team Dashboard")
    st.markdown(
        "Eight macro-financial relationships, each analysed under four "
        "supervising econometric principles and reported **side-by-side for the "
        "US and Canada** (where Canadian data exists). Use the tabs above to "
        "drill into any relationship; every tab highlights the three deliverables "
        "— **Coefficient**, **Lead-lag**, and **Chart** — per country."
    )

    st.subheader("Master summary")
    st.caption(
        "One row per relationship **per country**. Headline = honest "
        "contemporaneous correlation on the correct transform. "
        "🟥 red = sign flips across regimes / link collapses under control; "
        "🟩 green = stable / survives; 🟨 amber = auto-peak and theory-peak diverge."
    )
    st.dataframe(style_summary(SUMMARY), use_container_width=True,
                 hide_index=True)

    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("Data-asset catalogue")
        st.caption("Every series used, its coverage and stationarity classification.")
        if not CATALOGUE.empty:
            st.dataframe(CATALOGUE, use_container_width=True, hide_index=True)
        else:
            st.info("data_asset_summary.csv not found.")
    with c2:
        st.subheader("Output-gap sanity check")
        st.caption(
            "Derived gap = 100·(actual real GDP / CBO potential − 1). "
            "Must be **negative** in 2008-Q4 & 2020-Q2 (busts) and **positive** "
            "in 2021-Q4 (overheating) — confirming correct orientation."
        )
        gp = DATA_DIR / "output_gap_diagnostic.png"
        if gp.exists():
            st.image(str(gp), use_container_width=True)
        else:
            st.info("output_gap_diagnostic.png not found.")


# ====================  PER-RELATIONSHIP TABS  ============================== #
def render_panel(box, rel_name: str, country: str, compact: bool):
    """Render the three deliverables + diagnostics for ONE country into `box`.

    `box` is a Streamlit container (the tab itself, or one half of a 2-column
    split for side-by-side US|Canada). `compact` stacks the scatter/rolling
    charts vertically so they fit a half-width column.
    """
    n = rel_number(rel_name)
    cc_code = CC_CODE.get(country, "us")
    sel = SUMMARY[(SUMMARY["Relationship"] == rel_name)]
    if "Country" in sel.columns:
        sel = sel[sel["Country"] == country]
    if sel.empty:
        box.info(f"No {country} data for this relationship.")
        return
    row = sel.iloc[0]
    mx = metric_lookup(rel_name, country)

    contemp = mx.get("contemporaneous_r", row.get("Contemporaneous r"))
    theory_r = mx.get("theory_peak_r", row.get("Theory peak r"))
    theory_lag = mx.get("theory_peak_lag", row.get("Theory peak lag"))
    auto_r = mx.get("auto_peak_r", row.get("Auto peak r"))
    auto_lag = mx.get("auto_peak_lag", row.get("Auto peak lag"))
    n_obs = mx.get("n_obs")
    band = mx.get("sig_band_95")
    lead_var = row.get("Leading variable", "—")
    raw_pc = mx.get("raw_partial_corr")
    part_pc = mx.get("partial_corr")
    coint = row.get("Cointegrated?", "n/a")
    flip = row.get("Sign-flips across regimes?", "—")
    survive = row.get("Survives common-driver control?", "n/a")
    divergence = row.get("Auto/Theory divergence", "—")
    ukey = f"{n}_{cc_code}"

    box.markdown(f"### {CC_LABEL.get(country, country)}")
    # Country-specific methodology caveats (kept visible so the team isn't misled
    # by a shared relationship label).
    if n == 5 and country == "Canada":
        box.caption(
            "⚠️ Canada uses a **10Y-2Y** slope proxy (GoC 10Y − GoC 2Y); no GoC 3M "
            "series exists. The US panel uses the canonical 10Y-3M."
        )
    if n == 8 and country == "Canada":
        box.caption(
            "Uses the **global VIX** (no Canadian volatility analog) against "
            "Canadian GDP growth."
        )

    # ---- DELIVERABLE 1: COEFFICIENT ------------------------------------- #
    box.markdown("**① Coefficient**")
    sig_contemp = (band is not None and contemp is not None
                   and not np.isnan(float(contemp))
                   and abs(float(contemp)) > float(band))
    k1, k2 = box.columns(2)
    k1.metric("Headline r (contemporaneous)", fmt(contemp),
              help="Honest correlation on the ADF-correct transform.")
    k2.metric("Significant @95%?", "Yes ✓" if sig_contemp else "No",
              help=f"|r| vs ±{fmt(band)} band (±1.96/√n).")
    k3, k4 = box.columns(2)
    k3.metric("Theory-peak r", fmt(theory_r),
              help="Largest |r| consistent with the expected sign & direction.")
    k4.metric("Observations (n)",
              fmt(n_obs, nd=0) if n_obs is not None else "—")
    box.caption(f"Transform used: **{row.get('Transform', '—')}**")

    # ---- DELIVERABLE 2: LEAD-LAG ---------------------------------------- #
    box.markdown("**② Lead-lag**")
    l1, l2, l3 = box.columns(3)
    l1.metric("Theory peak lag (Q)",
              fmt(theory_lag, nd=0) if theory_lag is not None else "—")
    l2.metric("Leading variable", str(lead_var))
    l3.metric("Auto peak", f"{fmt(auto_r)} @ {fmt(auto_lag, nd=0)}Q")

    if str(divergence).strip() not in ("—", "", "nan"):
        box.warning(
            f"**Auto/theory divergence:** {divergence}  \n"
            "The largest-|r| lag disagrees with the theory-consistent one — "
            "shown transparently rather than hidden."
        )

    box.plotly_chart(
        build_ccf_figure(rel_name, auto_lag, theory_lag, band, country),
        use_container_width=True, key=f"ccf_{ukey}")
    box.caption(
        "Cross-correlation r(X leads Y by *lag*). **Positive lag = X leads Y.** "
        "Shaded band = ±95% significance. "
        "🔴 auto-peak　🟢 theory-peak."
    )

    # ---- DELIVERABLE 3: CHART ------------------------------------------- #
    box.markdown("**③ Chart**")
    sub = SERIES[SERIES["relationship"] == rel_name]
    if "country" in sub.columns:
        sub = sub[sub["country"] == country]
    sub = sub.dropna(subset=["x", "y"]).sort_values("date")
    ccf_c = CCF[CCF["relationship"] == rel_name]
    if "country" in ccf_c.columns:
        ccf_c = ccf_c[ccf_c["country"] == country]

    if not sub.empty and not ccf_c.empty:
        lag_min, lag_max = int(ccf_c["lag"].min()), int(ccf_c["lag"].max())
        default_lag = int(theory_lag) if (theory_lag is not None
            and not (isinstance(theory_lag, float) and np.isnan(theory_lag))) \
            else int(auto_lag or 0)
        default_lag = max(lag_min, min(lag_max, default_lag))

        lag = box.slider(
            "Scatter at lag (quarters, X leads Y →)",
            lag_min, lag_max, default_lag, key=f"lag_{ukey}",
            help="Shift X relative to Y to inspect the link at any lag. "
                 "Default = theory-peak lag.",
        )
        if compact:
            box.plotly_chart(build_scatter_figure(sub, lag, rel_name),
                             use_container_width=True, key=f"sc_{ukey}")
            box.plotly_chart(build_rolling_figure(rel_name, contemp, country),
                             use_container_width=True, key=f"roll_{ukey}")
        else:
            cL, cR = box.columns(2)
            cL.plotly_chart(build_scatter_figure(sub, lag, rel_name),
                            use_container_width=True, key=f"sc_{ukey}")
            cR.plotly_chart(build_rolling_figure(rel_name, contemp, country),
                            use_container_width=True, key=f"roll_{ukey}")
        box.caption(
            "12-quarter rolling correlation. Dashed grey = full-sample r; "
            "vertical red = detected structural breaks (Principle 3)."
        )
    else:
        box.info("Transformed-series data unavailable for the interactive chart.")

    fp = fig_path(n, cc_code)
    if fp:
        with box.expander("📄 Publication figure (static, 300 dpi)"):
            st.image(str(fp), use_container_width=True)
            st.download_button(
                f"⬇ {fp.name}", fp.read_bytes(), file_name=fp.name,
                mime="image/png", key=f"dlfig_{ukey}",
            )

    # ---- DELIVERABLE 5: TIME-VARYING OKUN SLOPE (Okun tab only) ---------- #
    if n == 1:
        render_okun_tvp(box, country)

    # ---- ALL OTHER DATA ------------------------------------------------- #
    box.markdown("**④ All other diagnostics**")
    d1, d2, d3 = box.columns(3)
    d1.metric("Cointegrated?", str(coint),
              help="Only valid when BOTH series are I(1) in levels; else n/a.")
    d2.metric("Sign flips across regimes?", str(flip),
              help="Does the rolling correlation change sign? (Principle 3)")
    d3.metric("Survives common-driver control?", str(survive),
              help="Does the link hold after partialling out a shared shock?")

    with box.expander("Principle 4 — raw vs partial correlation"):
        if raw_pc is not None and part_pc is not None:
            p1, p2 = st.columns(2)
            p1.metric("Raw correlation", fmt(raw_pc))
            p2.metric("Partial (controlled)", fmt(part_pc),
                      delta=fmt(float(part_pc) - float(raw_pc)))
            st.caption(
                "If the partial correlation collapses toward zero, the raw "
                "link was largely a shared-shock artefact; if it holds, the "
                "relationship is robust to the common driver."
            )
        else:
            st.caption("Partial-correlation control not applicable to this pair.")

    with box.expander("Structural breaks (detected break dates)"):
        bk = BREAKS[BREAKS["relationship"] == rel_name]
        if "country" in bk.columns:
            bk = bk[bk["country"] == country]
        if not bk.empty:
            quarters = (pd.to_datetime(bk["break_date"])
                        .dt.to_period("Q").astype(str).tolist())
            st.write(", ".join(quarters))
        else:
            st.caption("No structural breaks detected in the relationship.")

    with box.expander("All metrics for this relationship"):
        sub_m = DASH[DASH["relationship"] == rel_name]
        if "country" in sub_m.columns:
            sub_m = sub_m[sub_m["country"] == country]
        st.dataframe(sub_m[["metric_name", "metric_value"]],
                     use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Relationship 6 — Energy Prices & Inflation (Threshold VECM, US vs Canada)
# --------------------------------------------------------------------------- #
# Verified results. US numbers are hardcoded from estimate_tvecm.R / vecm_forecast.py;
# Canada numbers are from estimate_tvecm_canada.R (Hansen-Seo TVECM, nboot=5000).
TVECM_US = {
    "flag": "🇺🇸 United States", "energy_series": "BLS final-demand energy PPI (FPP)",
    "beta": "1.099", "elasticity": "~1.1", "model": "TVECM",
    "suplm": "43.84", "p_fixed": "0.000", "p_resid": "0.000",
    "gamma": "−0.4798", "verdict": "REJECTED linear",
    "turb_pct": "13%", "norm_pct": "87%",
    "turb_era": "Pre-2000 oil shocks (1973–74, 1979–80)",
    "oos_turb": "Jun–Nov 2022 (6 months)",
    "adj_burden": "Energy / FPP (α₂ significant, α₁ ≈ 0)",
    "cluster_p": "< 0.001",
    "h1_rmse": "+4.89%", "h1_dm": "0.073", "h1_sig": False,
    "h3_rmse": "+8.72%", "h3_dm": "0.037", "h3_sig": True,
    "corr": "0.796",
    "levels_img": "chart1_levels.png",
    "ect_img": "tvecm_ect_regimes.png",
    "regime_img": "regime_detection_final.png",
    "forecast_img": "vecm_forecast_final.png",
    "i1_note": "Both series cleanly I(1); Johansen confirms r = 1.",
}
TVECM_CA = {
    "flag": "🇨🇦 Canada", "energy_series": "reconstructed energy-inflation index (base 100)",
    "beta": "0.664", "elasticity": "~0.66", "model": "TVECM",
    "suplm": "66.89", "p_fixed": "0.000", "p_resid": "0.000",
    "gamma": "−0.0662", "verdict": "REJECTED linear",
    "turb_pct": "30.5%", "norm_pct": "69.5%",
    "turb_era": "1962–2008 (1970s–80s shocks + 2008); only 3 turbulent mo. post-2000",
    "oos_turb": "0 months (2022 surge stayed normal)",
    "adj_burden": "Energy (energy-eq α significant, p = 0.0003)",
    "cluster_p": "< 0.001",
    "h1_rmse": "+4.22%", "h1_dm": "0.221", "h1_sig": False,
    "h3_rmse": "+5.83%", "h3_dm": "0.104", "h3_sig": False,
    "corr": "0.666",
    "levels_img": "chart1_levels_canada.png",
    "ect_img": "tvecm_ect_regimes_canada.png",
    "regime_img": "regime_detection_canada.png",
    "forecast_img": "vecm_forecast_canada.png",
    "i1_note": ("log-energy I(1) confirmed; log-CPI borderline (differenced ADF near "
                "the 10% bound — NSA seasonality). Johansen rejects r = 0 at 1% (r = 1 "
                "confirmed); the 5% trace hints at r = 2, consistent with CPI's borderline "
                "integration. Cointegrating vector β ≈ 0.66 matches the TVECM."),
}

_TILE_PALETTE = {
    "neutral":   ("#f4f6f8", "#1f2933", "#d9dee3"),
    "blue":      ("#e7f0fb", "#1c4e8a", "#a6c8ef"),
    "normal":    ("#e3f6e5", "#1c6b2a", "#a6d8ad"),
    "turbulent": ("#fde7e7", "#8a1c12", "#e7a39c"),
    "gold":      ("#fff7e0", "#7a5400", "#e0b73a"),
}


def _tvecm_tile(box, label, value, kind="neutral", star=False, sub=None):
    bg, fg, border = _TILE_PALETTE.get(kind, _TILE_PALETTE["neutral"])
    bw = "3px" if kind == "gold" else "1px"
    star_html = " ★" if star else ""
    sub_html = (f"<div style='font-size:0.70rem;color:{fg};opacity:.72;margin-top:3px'>"
                f"{sub}</div>") if sub else ""
    box.markdown(
        f"<div style='background:{bg};border:{bw} solid {border};border-radius:10px;"
        f"padding:9px 12px;margin-bottom:9px;min-height:74px'>"
        f"<div style='font-size:0.70rem;color:{fg};opacity:.8;text-transform:uppercase;"
        f"letter-spacing:.03em;font-weight:600'>{label}{star_html}</div>"
        f"<div style='font-size:1.22rem;font-weight:700;color:{fg};margin-top:2px'>{value}</div>"
        f"{sub_html}</div>",
        unsafe_allow_html=True,
    )


def _tvecm_img(box, fname, caption=None):
    p = DATA_DIR / fname
    if p.exists():
        box.image(str(p), use_container_width=True, caption=caption)
    else:
        box.info(f"{fname} not found.")


def render_tvecm_tab(tab):
    """Energy↔inflation threshold-VECM tab: US (left) vs Canada (right)."""
    with tab:
        st.title("⚡ Energy Prices & Inflation (TVECM)")
        st.markdown(WHY.get(6, ""))
        st.caption(
            "Full **Hansen–Seo threshold vector error-correction** analysis. Both "
            "series are I(1) and cointegrated, the linear-cointegration null is "
            "rejected in favour of a two-regime threshold model, and the model is "
            "validated out-of-sample against an AR(2) benchmark — US left, Canada right."
        )
        st.divider()

        # ---- SECTION 1 — The Relationship -------------------------------- #
        st.header("1 · The Relationship")
        cL, cR = st.columns(2)
        for col, d in ((cL, TVECM_US), (cR, TVECM_CA)):
            col.subheader(d["flag"])
            a, b = col.columns(2)
            _tvecm_tile(a, "Cointegrating β", d["beta"], "blue")
            _tvecm_tile(b, "Long-run elasticity", d["elasticity"], "blue")
            a, b = col.columns(2)
            _tvecm_tile(a, "Model", d["model"], "neutral")
            _tvecm_tile(b, "Coefficient", "Significant", "normal")
            _tvecm_img(col, d["levels_img"], "Consumer prices vs energy prices (levels)")
        st.markdown(
            "Energy prices and consumer inflation share a **genuine long-run "
            "equilibrium** in both countries — a 1% permanent rise in energy prices is "
            "associated with roughly a **1.10%** long-run rise in **US** CPI and **0.66%** "
            "in **Canadian** CPI. US pass-through is essentially **one-for-one**, while "
            "Canada's is **partial (about two-thirds)** — plausible given Canada's energy "
            "mix (hydroelectric, regulated utilities) and the reconstructed energy proxy. "
            "Both are consistent with **Blanchard & Galí (2010)** and the broader "
            "energy-pass-through literature."
        )
        st.divider()

        # ---- SECTION 2 — Threshold Test & Cointegration ------------------ #
        st.header("2 · Threshold Test & Cointegration")
        cL, cR = st.columns(2)
        for col, d in ((cL, TVECM_US), (cR, TVECM_CA)):
            col.subheader(d["flag"])
            a, b, c = col.columns(3)
            _tvecm_tile(a, "SupLM statistic", d["suplm"], "neutral")
            _tvecm_tile(b, "Fixed-reg. bootstrap p", d["p_fixed"], "normal")
            _tvecm_tile(c, "Residual bootstrap p", d["p_resid"], "normal")
            a, b, c = col.columns(3)
            _tvecm_tile(a, "Threshold γ", d["gamma"], "neutral")
            _tvecm_tile(b, "Cointegrating β", d["beta"], "blue")
            _tvecm_tile(c, "Verdict", d["verdict"], "turbulent")
            col.caption(f"**Integration / rank:** {d['i1_note']}")
        st.markdown(
            "The **SupLM test decisively rejects linear cointegration** in both countries "
            "(both bootstrap p-values ≈ 0.000) in favour of the two-regime threshold model. "
            "The relationship does **not adjust at a constant speed** — it adjusts "
            "differently depending on how far the two series have drifted from their "
            "long-run equilibrium. Canada's SupLM (**66.89**) is in fact *larger* than the "
            "US (**43.84**), an even stronger rejection of linearity."
        )
        ec_L, ec_R = st.columns(2)
        _tvecm_img(ec_L, TVECM_US["ect_img"], "US — demeaned ECT & threshold")
        _tvecm_img(ec_R, TVECM_CA["ect_img"], "Canada — demeaned ECT & threshold")
        st.divider()

        # ---- SECTION 3 — Regime Classification --------------------------- #
        st.header("3 · Regime Classification")
        cL, cR = st.columns(2)
        for col, d in ((cL, TVECM_US), (cR, TVECM_CA)):
            col.subheader(d["flag"])
            a, b = col.columns(2)
            _tvecm_tile(a, "Turbulent %", d["turb_pct"], "turbulent")
            _tvecm_tile(b, "Normal %", d["norm_pct"], "normal")
            _tvecm_tile(col, "Turbulent era (historical)", d["turb_era"], "neutral")
            _tvecm_tile(col, "Out-of-sample turbulent", d["oos_turb"], "turbulent")
            _tvecm_tile(col, "Adjustment burden", d["adj_burden"], "neutral")
            _tvecm_tile(col, "Clustering test p-value", d["cluster_p"], "normal",
                        sub="Wald–Wolfowitz runs test on the regime sequence")
        st.markdown(
            "Both countries show the **same broad structural pattern** — the turbulent "
            "regime is concentrated in the **pre-2000/pre-2008 era** of large oil shocks "
            "and less-anchored inflation expectations, with the normal regime dominant in "
            "recent decades. A meaningful difference: **Canada spends far more of its "
            "history turbulent (30.5% vs 13%)**, reflecting greater commodity-price "
            "exposure (WCS crude, AECO gas). Yet the **2022 energy surge produced only a "
            "brief 6-month turbulent episode in the US and did *not* tip Canada into the "
            "turbulent regime at all** — Canadian CPI tracked energy closely enough to stay "
            "within the normal band — consistent with the **anchored inflation expectations "
            "of the modern monetary-policy era**. Both regime sequences are strongly "
            "**clustered (runs-test p < 0.001)**, confirming genuine persistence rather "
            "than random switching."
        )
        rc_L, rc_R = st.columns(2)
        _tvecm_img(rc_L, TVECM_US["regime_img"], "US — regime classification")
        _tvecm_img(rc_R, TVECM_CA["regime_img"], "Canada — regime classification")
        st.divider()

        # ---- SECTION 4 — Out-of-Sample Validation ------------------------ #
        st.header("4 · Out-of-Sample Validation")
        cL, cR = st.columns(2)
        for col, d in ((cL, TVECM_US), (cR, TVECM_CA)):
            col.subheader(d["flag"])
            a, b = col.columns(2)
            _tvecm_tile(a, "h = 1 · RMSE improvement", d["h1_rmse"], "neutral")
            _tvecm_tile(b, "h = 1 · DM p-value", d["h1_dm"], "neutral")
            a, b = col.columns(2)
            h3_kind = "gold" if d["h3_sig"] else "neutral"
            _tvecm_tile(a, "h = 3 · RMSE improvement", d["h3_rmse"], h3_kind,
                        star=d["h3_sig"])
            _tvecm_tile(b, "h = 3 · DM p-value", d["h3_dm"], h3_kind,
                        star=d["h3_sig"],
                        sub="significant at 5%" if d["h3_sig"] else "not significant")
        st.markdown(
            "The model was estimated on **pre-2022 data** and validated on genuinely unseen "
            "**2022–2026** data. In the **US the three-month-horizon gain is statistically "
            "significant** (RMSE −8.72%, DM p = 0.037 ★) — the key finding. **Canada shows a "
            "positive but smaller, statistically *insignificant* gain** (RMSE −5.83%, DM "
            "p = 0.104): the threshold model helps directionally but does not decisively "
            "beat a simple AR(2), partly because **Canada experienced no turbulent regime "
            "out-of-sample** — precisely the conditions under which the TVECM's "
            "regime-switching adds the most value. Both are consistent with an estimated "
            "energy-to-CPI pass-through timing of roughly **one to three months**."
        )
        fc_L, fc_R = st.columns(2)
        _tvecm_img(fc_L, TVECM_US["forecast_img"], "US — VECM vs AR(2) forecast")
        _tvecm_img(fc_R, TVECM_CA["forecast_img"], "Canada — VECM vs AR(2) forecast")
        with st.expander("US — RMSE improvement by horizon (h = 1…24)"):
            _tvecm_img(st, "forecast_by_horizon.png")
        st.divider()

        # ---- Footer + Canada proxy footnote ------------------------------ #
        st.info(
            "**Note:** The simple correlation (US r = 0.796, Canada r = 0.666) appears in "
            "the relationship leaderboard for comparability. This tab presents the **full "
            "structural analysis** justified by the data properties — both series I(1), "
            "cointegrated, and nonlinear in both countries. The **TVECM long-run elasticity "
            "is the economically meaningful coefficient**; the correlation is the comparable "
            "surface measure."
        )
        st.caption(
            "**Canadian energy proxy:** a base-100 energy price level reconstructed from the "
            "*'Energy Inflation'* (YoY %) series by 12-month chaining (no clean Canadian "
            "energy-PPI *level* exists in the source); the YoY-cumulation introduces a mild "
            "seam in the early sample. Note Canada's energy mix (**hydroelectric, AECO gas, "
            "WCS crude**) differs from the US, which may affect pass-through timing and "
            "magnitude — a plausible driver of the lower Canadian elasticity (β ≈ 0.66 vs US "
            "1.10). Canadian log-CPI is also only borderline I(1) (NSA seasonality), so the "
            "Canadian cointegration/TVECM results carry a slightly larger caveat than the US."
        )


def render_relationship(tab, rel_name: str):
    n = rel_number(rel_name)
    if n == 6:
        render_tvecm_tab(tab)
        return
    countries = countries_for(rel_name)
    with tab:
        st.title(rel_name)
        st.markdown(WHY.get(n, ""))
        if "Cross-border" in countries:
            st.info(
                "This is a **cross-border** relationship (US vs Canada by "
                "construction), so it is shown as a single panel rather than "
                "side-by-side."
            )
        st.divider()

        if len(countries) <= 1:
            render_panel(st, rel_name, countries[0] if countries else "US",
                         compact=False)
        else:
            cols = st.columns(len(countries))
            for col, ctry in zip(cols, countries):
                render_panel(col, rel_name, ctry, compact=True)


# --------------------------------------------------------------------------- #
# Plotly figure builders
# --------------------------------------------------------------------------- #
def build_ccf_figure(rel_name, auto_lag, theory_lag, band, country="US"):
    cc = CCF[CCF["relationship"] == rel_name]
    if "country" in cc.columns:
        cc = cc[cc["country"] == country]
    cc = cc.sort_values("lag")
    fig = go.Figure()
    if cc.empty:
        return fig

    fig.add_trace(go.Bar(
        x=cc["lag"], y=cc["r"], name="r",
        marker_color="#4c78a8",
        hovertemplate="lag %{x}Q<br>r = %{y:.3f}<extra></extra>",
    ))
    if band is not None and not np.isnan(float(band)):
        b = float(band)
        fig.add_hrect(y0=-b, y1=b, fillcolor="rgba(150,150,150,0.15)",
                      line_width=0)
        fig.add_hline(y=b, line=dict(color="grey", dash="dot", width=1))
        fig.add_hline(y=-b, line=dict(color="grey", dash="dot", width=1))
    fig.add_hline(y=0, line=dict(color="black", width=1))

    def mark(lag, colour, label):
        if lag is None or (isinstance(lag, float) and np.isnan(lag)):
            return
        hit = cc[cc["lag"] == int(lag)]
        if hit.empty:
            return
        fig.add_trace(go.Scatter(
            x=[int(lag)], y=[hit["r"].iloc[0]], mode="markers",
            marker=dict(color=colour, size=13, line=dict(color="white", width=1.5)),
            name=label,
            hovertemplate=f"{label}<br>lag %{{x}}Q<br>r = %{{y:.3f}}<extra></extra>",
        ))

    mark(auto_lag, "#e4572e", "Auto peak")
    mark(theory_lag, "#2a9d3c", "Theory peak")

    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="lag (quarters)  —  positive = X leads Y",
        yaxis_title="cross-correlation r",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        template="plotly_white",
    )
    return fig


def build_scatter_figure(sub, lag, rel_name):
    """Scatter of X(t) vs Y(t+lag): X leading Y by `lag` quarters."""
    s = sub.copy().reset_index(drop=True)
    x_label = s["x_label"].iloc[0] if "x_label" in s else "X"
    y_label = s["y_label"].iloc[0] if "y_label" in s else "Y"

    if lag >= 0:
        xv = s["x"].iloc[: len(s) - lag].to_numpy()
        yv = s["y"].iloc[lag:].to_numpy()
    else:
        k = -lag
        xv = s["x"].iloc[k:].to_numpy()
        yv = s["y"].iloc[: len(s) - k].to_numpy()

    mask = np.isfinite(xv) & np.isfinite(yv)
    xv, yv = xv[mask], yv[mask]

    fig = go.Figure()
    if len(xv) < 3:
        fig.update_layout(height=360, title="Insufficient overlap at this lag")
        return fig

    r = float(np.corrcoef(xv, yv)[0, 1])
    fig.add_trace(go.Scatter(
        x=xv, y=yv, mode="markers",
        marker=dict(color="#4c78a8", size=7, opacity=0.7),
        name="observations",
        hovertemplate="x %{x:.2f}<br>y %{y:.2f}<extra></extra>",
    ))
    slope, intercept = np.polyfit(xv, yv, 1)
    xs = np.linspace(xv.min(), xv.max(), 100)
    fig.add_trace(go.Scatter(
        x=xs, y=slope * xs + intercept, mode="lines",
        line=dict(color="#e4572e", width=2.5), name="OLS fit",
        hoverinfo="skip",
    ))
    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.98, showarrow=False,
        align="left", bgcolor="rgba(255,255,255,0.8)",
        text=(f"<b>r = {r:.3f}</b>　n = {len(xv)}<br>"
              f"slope = {slope:.3f}　lag = {lag}Q"),
    )
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=30, b=10),
        title=f"Scatter at lag {lag}Q (X leads Y)",
        xaxis_title=x_label, yaxis_title=y_label,
        template="plotly_white", showlegend=False,
    )
    return fig


def build_rolling_figure(rel_name, full_r, country="US"):
    roll = ROLLING[ROLLING["relationship"] == rel_name]
    if "country" in roll.columns:
        roll = roll[roll["country"] == country]
    roll = roll.sort_values("date")
    fig = go.Figure()
    if roll.empty:
        fig.update_layout(height=360, title="No rolling-correlation data")
        return fig

    fig.add_trace(go.Scatter(
        x=roll["date"], y=roll["rolling_corr"], mode="lines",
        line=dict(color="#4c78a8", width=2), name="rolling r",
        hovertemplate="%{x|%Y-%m}<br>r = %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color="black", width=1))
    if full_r is not None and not (isinstance(full_r, float) and np.isnan(full_r)):
        fig.add_hline(y=float(full_r),
                      line=dict(color="grey", dash="dash", width=1.5),
                      annotation_text=f"full-sample r = {float(full_r):.2f}",
                      annotation_position="top left")

    bk = BREAKS[BREAKS["relationship"] == rel_name]
    if "country" in bk.columns:
        bk = bk[bk["country"] == country]
    for _, brow in bk.iterrows():
        bd = brow["break_date"]
        if pd.notna(bd) and roll["date"].min() <= bd <= roll["date"].max():
            fig.add_vline(x=bd, line=dict(color="#e4572e", dash="dot", width=1.5))

    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=30, b=10),
        title="Rolling correlation (12-quarter window)",
        xaxis_title="date", yaxis_title="rolling r",
        yaxis=dict(range=[-1, 1]), template="plotly_white",
    )
    return fig


def build_okun_tvp_figure(country="US"):
    """Time-varying Okun slope (Beaton TVP / Kalman smoother) with 95% band."""
    fig = go.Figure()
    if OKUN_TVP.empty or "country" not in OKUN_TVP.columns:
        fig.update_layout(height=380, title="No time-varying Okun data")
        return fig
    d = OKUN_TVP[OKUN_TVP["country"] == country].sort_values("date")
    if d.empty:
        fig.update_layout(height=380, title=f"No time-varying Okun data for {country}")
        return fig

    # 95% band (upper then lower, filled).
    fig.add_trace(go.Scatter(
        x=d["date"], y=d["upper"], mode="lines",
        line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=d["date"], y=d["lower"], mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(76,120,168,0.18)",
        name="95% CI", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=d["date"], y=d["total_okun"], mode="lines",
        line=dict(color="#4c78a8", width=2.5),
        name="smoothed total slope α₁+α₂",
        hovertemplate="%{x|%Y-%m}<br>slope = %{y:+.3f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color="black", width=1))

    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=34, b=10),
        title=f"Time-varying Okun slope — {country}",
        xaxis_title="date", yaxis_title="α₁ + α₂",
        template="plotly_white", legend=dict(orientation="h", y=-0.18),
    )
    return fig


def render_okun_tvp(box, country):
    """Render the time-varying Okun slope band chart + regime table for one country."""
    if OKUN_TVP.empty or "country" not in OKUN_TVP.columns:
        return
    d = OKUN_TVP[OKUN_TVP["country"] == country]
    if d.empty:
        return
    cc = CC_CODE.get(country, "us")
    box.markdown("**⑤ Time-varying slope (Beaton TVP / Kalman smoother)**")
    box.caption(
        "Random-walk Okun coefficients estimated with a Stock–Watson (1998) "
        "median-unbiased state variance, a Lenza–Primiceri (2022) COVID "
        "down-weighting, and an RTS Kalman smoother. Shaded = 95% pointwise CI. "
        "**The key insight a static correlation misses: the slope drifts across "
        "regimes (Principle 3).**"
    )
    box.plotly_chart(build_okun_tvp_figure(country),
                     use_container_width=True, key=f"tvp_{cc}")
    smin = float(d["total_okun"].min()); smax = float(d["total_okun"].max())
    if smin < 0 < smax:
        box.warning(
            "⚠️ The smoothed Okun slope **changes sign** across the sample — "
            "Okun's law is *not* a stable constant here."
        )
    if not OKUN_TVP_REG.empty:
        reg = OKUN_TVP_REG[OKUN_TVP_REG["country"] == country]
        if not reg.empty:
            with box.expander("Per-regime averages of the smoothed slope"):
                show = reg[["regime", "mean", "sem", "n"]].copy()
                show["mean"] = show["mean"].map(lambda v: f"{float(v):+.3f}")
                show["sem"] = show["sem"].map(
                    lambda v: f"{float(v):.3f}" if pd.notna(v) else "—")
                st.dataframe(show, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# Render every relationship tab
# --------------------------------------------------------------------------- #
for i, rel in enumerate(REL_NAMES, start=1):
    render_relationship(tabs[i], rel)
