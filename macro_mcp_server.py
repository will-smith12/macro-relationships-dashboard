#!/usr/bin/env python3
"""
Macro-Financial Relationships — MCP server
==========================================

A local Model Context Protocol (MCP) server that lets Claude Desktop query the
results of the macro-financial relationship analysis *live* and pull exact
numbers on demand while chatting.

The Jupyter notebook ``macro_relationships_master.ipynb`` is the single COMPUTE
layer; it exports a set of CSVs. This server is a thin READ-ONLY query layer over
those CSVs — Claude calls its tools to fetch the master summary, a single
relationship's full dossier, the cross-correlation function, the rolling
correlation, structural breaks, the data catalogue, or to search any metric.

Transport: stdio (Claude Desktop launches this as a subprocess).

Run standalone (for debugging):
    python macro_mcp_server.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from mcp.server.fastmcp import FastMCP

# --------------------------------------------------------------------------- #
# Configuration & data loading
# --------------------------------------------------------------------------- #
DATA_DIR = Path(__file__).resolve().parent

mcp = FastMCP("macro-relationships")


def _read(name: str, keep_na_strings: bool = False) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        return pd.DataFrame()
    try:
        # keep_na_strings=True keeps literal "n/a" cells as text (not NaN), which
        # matters for the categorical summary/catalogue columns.
        return pd.read_csv(path, keep_default_na=not keep_na_strings)
    except Exception:
        return pd.DataFrame()


# Loaded once at import; the analysis artefacts are static between notebook runs.
SUMMARY = _read("relationship_summary.csv", keep_na_strings=True)
DASH = _read("dashboard_data.csv")
CCF = _read("ccf_data.csv")
ROLLING = _read("rolling_correlations.csv")
BREAKS = _read("breaks_data.csv")
CATALOGUE = _read("data_asset_summary.csv", keep_na_strings=True)
OKUN_TVP = _read("okun_tvp.csv")
OKUN_TVP_REG = _read("okun_tvp_regimes.csv")

REL_NAMES: list[str] = (
    list(dict.fromkeys(SUMMARY["Relationship"].tolist())) if not SUMMARY.empty else []
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _rel_number(name: str) -> int:
    try:
        return int(str(name).split("·")[0].strip().split(" ")[0])
    except Exception:
        return 0


# Keywords -> relationship number, for fuzzy resolution.
_KEYWORDS = {
    1: ["okun", "unemployment vs gdp", "okun's law"],
    2: ["phillips", "unemployment vs cpi", "unemployment vs inflation"],
    3: ["output gap", "gap vs cpi", "gap vs inflation", "potential"],
    4: ["taylor", "inflation vs rate", "policy rate", "cpi vs policy", "fed funds"],
    5: ["yield", "slope", "curve", "10y-3m", "10y3m", "recession", "spread vs gdp"],
    6: ["energy", "oil", "wti", "energy vs cpi", "pass-through", "passthrough"],
    7: ["fx", "exchange", "currency", "ir diff", "differential", "usd", "cad"],
    8: ["vix", "volatility", "uncertainty", "risk", "bloom", "fear", "market stress"],
}


def resolve(name_or_number) -> str | None:
    """Map '3', 3, 'Okun', 'output gap', a full label, etc. -> canonical name."""
    if not REL_NAMES:
        return None
    s = str(name_or_number).strip().lower()

    # Exact full label.
    for rn in REL_NAMES:
        if s == rn.lower():
            return rn
    # Leading number ("3", "rel 3", "#3", "3 · ...").
    digits = "".join(ch for ch in s.split("·")[0] if ch.isdigit())
    if digits:
        num = int(digits)
        for rn in REL_NAMES:
            if _rel_number(rn) == num:
                return rn
    # Keyword match.
    for num, kws in _KEYWORDS.items():
        if any(kw in s for kw in kws):
            for rn in REL_NAMES:
                if _rel_number(rn) == num:
                    return rn
    # Loose substring on the label.
    for rn in REL_NAMES:
        core = rn.lower().split("·", 1)[-1]
        if s and s in core:
            return rn
    return None


def _metrics(rel: str, country: str | None = None) -> dict:
    if DASH.empty:
        return {}
    sub = DASH[DASH["relationship"] == rel]
    if country is not None and "country" in sub.columns:
        sub = sub[sub["country"] == country]
    return dict(zip(sub["metric_name"], sub["metric_value"]))


# Country resolution (US / Canada / Cross-border) ----------------------------- #
_COUNTRY_CANON = {
    "us": "US", "u.s.": "US", "usa": "US", "united states": "US", "america": "US",
    "ca": "Canada", "can": "Canada", "canada": "Canada", "canadian": "Canada",
    "xb": "Cross-border", "cross-border": "Cross-border", "cross border": "Cross-border",
    "crossborder": "Cross-border",
}


def resolve_country(country) -> str:
    if country is None:
        return "US"
    return _COUNTRY_CANON.get(str(country).strip().lower(), str(country).strip())


def _countries_for(rel: str) -> list[str]:
    if SUMMARY.empty or "Country" not in SUMMARY.columns:
        return ["US"]
    return SUMMARY[SUMMARY["Relationship"] == rel]["Country"].unique().tolist()


def _pick_country(rel: str, requested) -> tuple[str, str | None]:
    """Return (country_to_use, note). Falls back to the only/first available
    country when the requested one is not present (e.g. asking for 'Canada' on the
    cross-border FX relationship)."""
    avail = _countries_for(rel)
    want = resolve_country(requested)
    if want in avail:
        return want, None
    use = avail[0]
    note = (f"_(note: {use} shown — '{requested}' is not available for this "
            f"relationship; available: {', '.join(avail)})_")
    return use, note


def _summary_row(rel: str, country: str | None = None) -> dict:
    row = SUMMARY[SUMMARY["Relationship"] == rel]
    if country is not None and "Country" in row.columns:
        row = row[row["Country"] == country]
    return row.iloc[0].to_dict() if not row.empty else {}


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no data)_"
    try:
        return df.to_markdown(index=False)
    except Exception:
        # to_markdown needs `tabulate`; fall back to a plain fixed-width table.
        return df.to_string(index=False)


def _quarter(date_str) -> str:
    try:
        return str(pd.Timestamp(date_str).to_period("Q"))
    except Exception:
        return str(date_str)


def _no_data_msg() -> str:
    return (
        "No analysis artefacts were found next to the server "
        f"({DATA_DIR}). Run the notebook first:\n\n"
        "    jupyter nbconvert --to notebook --execute --inplace "
        "macro_relationships_master.ipynb"
    )


# --------------------------------------------------------------------------- #
# Tools
# --------------------------------------------------------------------------- #
@mcp.tool()
def list_relationships() -> str:
    """List the eight macro-financial relationships available to query (number,
    full name, and which countries — US / Canada / Cross-border — are available).
    Call this first to discover what can be asked about. Most relationships have
    both US and Canada results; pass country='Canada' to the other tools for the
    Canadian figures."""
    if not REL_NAMES:
        return _no_data_msg()
    lines = []
    for n in REL_NAMES:
        cty = ", ".join(_countries_for(n))
        lines.append(f"{_rel_number(n)}. {n}  [{cty}]")
    return ("Available relationships (with countries):\n" + "\n".join(lines)
            + "\n\nMost tools accept an optional country argument "
              "('US' default, or 'Canada').")


@mcp.tool()
def get_methodology() -> str:
    """Explain the four supervising econometric principles and the ADF-driven
    transform routing that every reported number respects. Use this to ground any
    explanation of why a given figure was computed the way it was."""
    return (
        "# Methodology — four supervising principles\n\n"
        "Every relationship is analysed on quarterly US (and US–Canada) macro data. "
        "Each series is ADF-tested and routed to the correct transform before any "
        "correlation is reported.\n\n"
        "**Principle 1 — Non-stationarity / avoid spurious correlation.** Each series "
        "gets an Augmented Dickey-Fuller test and is classified I(0) (stationary) or "
        "I(1)/I(1+) (needs differencing). If both series are I(1) we correlate their "
        "growth rates / first differences (the honest figure) and run a Johansen "
        "cointegration test for a genuine long-run link. A levels correlation is never "
        "the headline. Cointegration is reported ONLY when BOTH series are I(1) in "
        "levels; otherwise it is 'n/a'.\n\n"
        "**Principle 2 — Lead/lag over contemporaneous.** For every pair we compute the "
        "cross-correlation function over ±12 quarters. Convention: POSITIVE lag = X "
        "leads Y. We report the contemporaneous r, the AUTO peak (largest |r|), and the "
        "THEORY peak (largest |r| that matches the economically expected SIGN and "
        "leading DIRECTION). Where auto and theory peaks diverge we flag it explicitly "
        "rather than hide it.\n\n"
        "**Principle 3 — Regime-dependence / structural breaks.** We do not assume one "
        "stable coefficient. We detect structural breaks (ruptures) and compute a "
        "12-quarter ROLLING correlation so regime shifts and SIGN FLIPS are visible, "
        "reporting per-regime correlations.\n\n"
        "**Principle 4 — Common-driver screen.** Many of these variables share shocks "
        "(an oil spike moves energy, inflation AND rates). For the at-risk pairs we "
        "compute a PARTIAL correlation controlling for a common driver (energy or the "
        "policy rate) and report raw vs partial side by side — does the link survive or "
        "collapse?"
    )


@mcp.tool()
def get_master_summary() -> str:
    """Return the master summary table — one row per relationship with the headline
    correlation, auto/theory peak lead-lag, cointegration verdict, sign-flip flag and
    common-driver-survival flag. Best for a high-level overview or comparison."""
    if SUMMARY.empty:
        return _no_data_msg()
    return "# Master summary\n\n" + _df_to_markdown(SUMMARY)


@mcp.tool()
def get_relationship(relationship: str | int, country: str = "US") -> str:
    """Full dossier for ONE relationship (for one country): transform used,
    headline/contemporaneous correlation and 95% significance, auto-peak and
    theory-peak lead/lag (with any divergence note), leading variable,
    cointegration (only if valid), regime sign-flip, raw-vs-partial common-driver
    control, structural breaks and n.

    `relationship` accepts a number ('3'), a keyword ('Okun', 'output gap', 'yield
    slope', 'FX') or the full label. `country` accepts 'US' (default) or 'Canada'
    (the cross-border FX relationship is US↔Canada by construction)."""
    if SUMMARY.empty:
        return _no_data_msg()
    rel = resolve(relationship)
    if rel is None:
        return (f"Could not match '{relationship}'. "
                f"Call list_relationships to see valid options.")
    ctry, cnote = _pick_country(rel, country)

    row = _summary_row(rel, ctry)
    m = _metrics(rel, ctry)

    def g(key, default="—"):
        v = m.get(key)
        if v is None:
            return default
        try:
            f = float(v)
            return f"{f:.3f}" if not (f != f) else default  # nan check
        except (TypeError, ValueError):
            return str(v)

    def gi(key, default="—"):
        """Integer-formatted metric (for counts and lags)."""
        v = m.get(key)
        if v is None:
            return default
        try:
            f = float(v)
            return str(int(round(f))) if not (f != f) else default
        except (TypeError, ValueError):
            return str(v)

    def cell(key, default="n/a"):
        """A categorical summary cell, mapping NaN/empty to a clean label."""
        v = row.get(key)
        if v is None:
            return default
        s = str(v).strip()
        return default if s.lower() in ("nan", "", "none") else s

    band = m.get("sig_band_95")
    contemp = m.get("contemporaneous_r")
    try:
        sig = (band is not None and contemp is not None
               and abs(float(contemp)) > float(band))
    except (TypeError, ValueError):
        sig = False

    bk = BREAKS[BREAKS["relationship"] == rel] if not BREAKS.empty else pd.DataFrame()
    if not bk.empty and "country" in bk.columns:
        bk = bk[bk["country"] == ctry]
    break_q = [_quarter(d) for d in bk["break_date"]] if not bk.empty else []

    out = [
        f"# {rel}  — {ctry}",
        "",
        f"- **Transform used:** {cell('Transform', '—')}",
        f"- **Headline (contemporaneous) r:** {g('contemporaneous_r')}"
        f"  ({'significant' if sig else 'not significant'} at 95%, "
        f"band ±{g('sig_band_95')}, n = {gi('n_obs')})",
        f"- **Auto peak (largest |r|):** r = {g('auto_peak_r')} at lag "
        f"{gi('auto_peak_lag')} quarters",
        f"- **Theory peak (sign/direction-consistent):** r = {g('theory_peak_r')} "
        f"at lag {gi('theory_peak_lag')} quarters",
        f"- **Leading variable:** {cell('Leading variable', '—')}  "
        f"(positive lag = first variable leads)",
        f"- **Cointegrated?** {cell('Cointegrated?')}  "
        f"(only meaningful when both series are I(1) in levels)",
        f"- **Sign flips across regimes?** {cell('Sign-flips across regimes?', '—')}",
        f"- **Survives common-driver control?** "
        f"{cell('Survives common-driver control?')}  "
        f"(raw r = {g('raw_partial_corr')} → partial r = {g('partial_corr')})",
        f"- **Structural breaks:** {', '.join(break_q) if break_q else 'none detected'}",
    ]
    div = row.get("Auto/Theory divergence", "")
    if div and str(div).strip() not in ("—", "", "nan"):
        out += ["", f"⚠️ **Auto/theory divergence:** {div} — the largest-|r| lag "
                "disagrees with the theory-consistent one (shown transparently)."]
    if cnote:
        out += ["", cnote]
    return "\n".join(out)


@mcp.tool()
def get_cross_correlation(relationship: str | int, country: str = "US") -> str:
    """Return the full cross-correlation function for a relationship (for one
    country): r at every lag from -12 to +12 quarters, the ±95% significance band,
    and which lags are significant. Positive lag = the first variable leads. Useful
    for explaining the lead/lag structure in detail. `country` = 'US' (default) or
    'Canada'."""
    if CCF.empty:
        return _no_data_msg()
    rel = resolve(relationship)
    if rel is None:
        return f"Could not match '{relationship}'. Call list_relationships."
    ctry, cnote = _pick_country(rel, country)
    sub = CCF[CCF["relationship"] == rel]
    if "country" in sub.columns:
        sub = sub[sub["country"] == ctry]
    sub = sub.sort_values("lag")
    if sub.empty:
        return f"No cross-correlation data for '{rel}'."
    band = float(sub["band"].iloc[0]) if "band" in sub else float("nan")
    peak = sub.loc[sub["r"].abs().idxmax()]
    contemp = sub[sub["lag"] == 0]
    contemp_r = float(contemp["r"].iloc[0]) if not contemp.empty else float("nan")
    tbl = sub[["lag", "r", "sig"]].copy()
    head = (
        f"# Cross-correlation — {rel} ({ctry})\n\n"
        f"- ±95% significance band: ±{band:.3f}\n"
        f"- Contemporaneous (lag 0) r: {contemp_r:.3f}\n"
        f"- Peak |r|: {float(peak['r']):.3f} at lag {int(peak['lag'])} quarters\n"
        f"- Convention: POSITIVE lag = the first variable leads the second.\n"
        + (f"\n{cnote}\n" if cnote else "") + "\n"
    )
    return head + _df_to_markdown(tbl)


@mcp.tool()
def get_rolling_correlation(relationship: str | int, country: str = "US",
                            max_points: int = 40) -> str:
    """Return the 12-quarter rolling correlation over time for a relationship (for
    one country), so you can describe regime shifts and sign flips (Principle 3).
    `country` = 'US' (default) or 'Canada'. The series is downsampled to at most
    `max_points` rows for readability."""
    if ROLLING.empty:
        return _no_data_msg()
    rel = resolve(relationship)
    if rel is None:
        return f"Could not match '{relationship}'. Call list_relationships."
    ctry, cnote = _pick_country(rel, country)
    sub = ROLLING[ROLLING["relationship"] == rel].copy()
    if "country" in sub.columns:
        sub = sub[sub["country"] == ctry]
    if sub.empty:
        return f"No rolling-correlation data for '{rel}'."
    sub = sub.dropna(subset=["rolling_corr"]).reset_index(drop=True)
    n = len(sub)
    if n > max_points:
        step = max(1, n // max_points)
        sub = sub.iloc[::step]
    rng = (f"ranges from {sub['rolling_corr'].min():.2f} to "
           f"{sub['rolling_corr'].max():.2f}")
    sub = sub.copy()
    sub["quarter"] = [_quarter(d) for d in sub["date"]]
    tbl = sub[["quarter", "rolling_corr"]]
    return (f"# Rolling correlation (12-quarter window) — {rel} ({ctry})\n\n"
            f"The rolling correlation {rng} across the sample "
            f"({n} windows total).\n\n"
            + (f"{cnote}\n\n" if cnote else "") + _df_to_markdown(tbl))


@mcp.tool()
def get_structural_breaks(relationship: str | int, country: str = "US") -> str:
    """Return the structural-break dates detected in a relationship (Principle 3)
    for one country. `country` = 'US' (default) or 'Canada'."""
    if BREAKS.empty and SUMMARY.empty:
        return _no_data_msg()
    rel = resolve(relationship)
    if rel is None:
        return f"Could not match '{relationship}'. Call list_relationships."
    ctry, cnote = _pick_country(rel, country)
    sub = BREAKS[BREAKS["relationship"] == rel] if not BREAKS.empty else pd.DataFrame()
    if not sub.empty and "country" in sub.columns:
        sub = sub[sub["country"] == ctry]
    if sub.empty:
        return f"No structural breaks detected in '{rel}' ({ctry})."
    qs = [_quarter(d) for d in sub["break_date"]]
    tail = f"  {cnote}" if cnote else ""
    return f"Structural breaks in '{rel}' ({ctry}): {', '.join(qs)}{tail}"


@mcp.tool()
def get_okun_tvp(country: str = "US", max_points: int = 30) -> str:
    """Return the TIME-VARYING Okun slope (Beaton-style TVP / Kalman smoother) for a
    country — the smoothed total slope alpha1+alpha2 with its 95% band over time,
    plus per-regime averages. This is the regime-dependence (Principle 3) view that
    a single static correlation hides. `country` = 'US' (default) or 'Canada'.
    Method: Stock-Watson (1998) median-unbiased state variance + Lenza-Primiceri
    (2022) COVID down-weighting + RTS smoother (ported from the cad-energy-inflation
    research project)."""
    if OKUN_TVP.empty:
        return ("No time-varying Okun data found. Re-run the notebook to generate "
                "okun_tvp.csv / okun_tvp_regimes.csv.")
    ctry = resolve_country(country)
    sub = OKUN_TVP[OKUN_TVP["country"] == ctry].copy()
    if sub.empty:
        avail = ", ".join(sorted(OKUN_TVP["country"].unique()))
        return f"No time-varying Okun slope for '{country}'. Available: {avail}."
    sub = sub.sort_values("date").reset_index(drop=True)
    smin, smax = float(sub["total_okun"].min()), float(sub["total_okun"].max())
    start_v = float(sub["total_okun"].iloc[0])
    end_v = float(sub["total_okun"].iloc[-1])
    sign_note = ""
    if smin < 0 < smax:
        sign_note = ("\n\n**The slope CHANGES SIGN across the sample — Okun's law "
                     "is not a stable constant for this country.**")

    n = len(sub)
    if n > max_points:
        step = max(1, n // max_points)
        sub_show = sub.iloc[::step].copy()
    else:
        sub_show = sub.copy()
    sub_show["quarter"] = [_quarter(d) for d in sub_show["date"]]
    sub_show["slope"] = sub_show["total_okun"].map(lambda v: f"{float(v):+.3f}")
    sub_show["95% CI"] = [f"[{float(l):+.3f}, {float(h):+.3f}]"
                          for l, h in zip(sub_show["lower"], sub_show["upper"])]
    path_tbl = _df_to_markdown(sub_show[["quarter", "slope", "95% CI"]])

    reg_md = ""
    if not OKUN_TVP_REG.empty:
        reg = OKUN_TVP_REG[OKUN_TVP_REG["country"] == ctry].copy()
        if not reg.empty:
            reg["mean"] = reg["mean"].map(lambda v: f"{float(v):+.3f}")
            reg["sem"] = reg["sem"].map(
                lambda v: f"{float(v):.3f}" if pd.notna(v) else "—")
            reg_md = ("\n\n## Per-regime averages of the smoothed slope\n\n"
                      + _df_to_markdown(reg[["regime", "mean", "sem", "n"]]))

    return (f"# Time-varying Okun slope — {ctry}\n\n"
            f"Smoothed total slope (alpha1+alpha2) runs from {start_v:+.3f} "
            f"(start) to {end_v:+.3f} (latest); range [{smin:+.3f}, {smax:+.3f}] "
            f"over {n} quarters.{sign_note}\n\n"
            f"## Slope path (downsampled to {len(sub_show)} points)\n\n"
            f"{path_tbl}{reg_md}")


@mcp.tool()
def get_data_catalogue() -> str:
    """Return the data-asset catalogue: every underlying series, its source, date
    range, frequency, observation count and ADF stationarity classification."""
    if CATALOGUE.empty:
        return _no_data_msg()
    return "# Data-asset catalogue\n\n" + _df_to_markdown(CATALOGUE)


@mcp.tool()
def search_metrics(query: str) -> str:
    """Search every computed metric (the long dashboard_data table) for rows whose
    relationship name or metric name contains `query` (case-insensitive). Handy for
    pulling a specific number like 'partial_corr' or all metrics mentioning 'peak'."""
    if DASH.empty:
        return _no_data_msg()
    q = str(query).strip().lower()
    mask = (DASH["relationship"].str.lower().str.contains(q, na=False)
            | DASH["metric_name"].str.lower().str.contains(q, na=False))
    hits = DASH[mask]
    if hits.empty:
        return f"No metrics matched '{query}'."
    return f"# Metrics matching '{query}'\n\n" + _df_to_markdown(hits)


# --------------------------------------------------------------------------- #
# Resources — expose the publication figures so Claude can reference them
# --------------------------------------------------------------------------- #
@mcp.resource("figure://{name}")
def figure(name: str) -> str:
    """Return the absolute path to a saved figure PNG (e.g. 'fig_us_2_unemployment_vs_cpi',
    'fig_ca_1_okun', or 'output_gap_diagnostic'). Figures are country-prefixed:
    fig_us_* (US), fig_ca_* (Canada), fig_xb_* (cross-border). Returns the file path
    so it can be opened/attached."""
    candidates = list(DATA_DIR.glob(f"{name}*.png"))
    if not candidates:
        return f"No figure found matching '{name}'."
    return str(candidates[0])


if __name__ == "__main__":
    mcp.run()
