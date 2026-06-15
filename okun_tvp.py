"""Time-varying Okun's law (Beaton 2010 + Stock-Watson 1998 + Lenza-Primiceri 2022).

Self-contained port of the time-varying-parameter (TVP) Okun estimator used in the
author's separate ``cad-energy-inflation`` research project, adapted for the
``macro_relationships`` workbook (US + Canada). All network / StatCan dependencies
have been stripped: the panel is built directly from two in-memory series
(unemployment rate level + GDP growth) that already exist in the master workbook.

The Beaton distributed-lag specification is

.. math:: \\dot u_t = \\alpha_0 + \\alpha_1 \\dot y_t
                 + \\alpha_2 \\dot y_{t-1} + \\varepsilon_t

with quarterly :math:`\\dot u_t` the first difference of the unemployment rate (pp)
and :math:`\\dot y_t` GDP growth. The total time-varying Okun slope is
:math:`\\alpha_{1,t} + \\alpha_{2,t}`, recovered with a Rauch-Tung-Striebel (RTS)
Kalman smoother whose state-innovation covariance is pinned by the Stock-Watson
(1998) median-unbiased estimator (inverting an Andrews 1993 QLRT), with a
Lenza-Primiceri (2022) time-varying observation variance to down-weight the 2020
COVID outliers while keeping them in-sample.

References
----------
* Andrews, D. W. K. (1993). "Tests for parameter instability and structural change
  with unknown change point." *Econometrica* 61(4), 821-856.
* Beaton, K. (2010). "Time-varying estimates of Okun's law for Canada." Bank of
  Canada Staff Working Paper 2010-7.
* Lenza, M. and Primiceri, G. E. (2022). "How to estimate a vector autoregression
  after March 2020." *Journal of Applied Econometrics* 37(4).
* Stock, J. H. and Watson, M. W. (1998). "Median unbiased estimation of coefficient
  variance in a time-varying parameter model." *JASA* 93(441).

Mathematics are faithfully reproduced from the original ``src/okun.py``; only the
data-ingestion layer differs.
"""
from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover - statsmodels optional for OLS baseline
    sm = None
from scipy.optimize import minimize


# ---------------------------------------------------------------------
# Stock-Watson (1998) Table 3, QLR column, X_t = 1, D = 1.
# These values ARE the authoritative source; do not "recompute" them.
# ---------------------------------------------------------------------
STOCK_WATSON_TABLE_3_QLR: Dict[int, float] = {
    0:  3.198,  1:  3.416,  2:  3.594,  3:  4.106,  4:  4.848,
    5:  5.689,  6:  6.682,  7:  7.626,  8:  9.160,  9: 10.660,
    10: 11.841, 11: 13.098, 12: 15.451, 13: 17.094, 14: 19.423,
    15: 21.682, 16: 23.342, 17: 24.920, 18: 28.174, 19: 30.736,
    20: 33.313, 21: 36.109, 22: 39.673, 23: 41.955, 24: 45.056,
    25: 48.647, 26: 50.983, 27: 55.514, 28: 59.278, 29: 61.311,
    30: 64.016,
}

# Generic macro regimes; empty windows are skipped by ``regime_averages``.
REGIME_WINDOWS: Dict[str, Tuple[pd.Period, pd.Period]] = {
    "pre_1990":         (pd.Period("1960Q1"), pd.Period("1989Q4")),
    "great_moderation": (pd.Period("1990Q1"), pd.Period("2008Q3")),
    "gfc_to_covid":     (pd.Period("2010Q1"), pd.Period("2019Q4")),
    "post_covid":       (pd.Period("2020Q1"), pd.Period("2099Q4")),
}


# ---------------------------------------------------------------------
# Panel construction (workbook-fed; no network)
# ---------------------------------------------------------------------
def build_panel(unemployment: pd.Series, gdp_growth: pd.Series) -> pd.DataFrame:
    """Build the quarterly Beaton Okun panel from two workbook series.

    Parameters
    ----------
    unemployment : pandas.Series
        Unemployment-rate LEVEL (percent), quarterly, datetime/period index.
    gdp_growth : pandas.Series
        GDP growth rate (percent), quarterly, aligned index. Used as-is for
        :math:`\\dot y_t` (consistent with the existing Okun §2.1 regressor).

    Returns
    -------
    pandas.DataFrame
        Quarterly ``PeriodIndex`` with columns ``gdp_growth_qq``,
        ``unemp_diff``, ``gdp_growth_lag1`` (NA rows dropped).
    """
    u = pd.Series(unemployment).astype(float).copy()
    g = pd.Series(gdp_growth).astype(float).copy()
    u.index = pd.PeriodIndex(pd.to_datetime(u.index), freq="Q")
    g.index = pd.PeriodIndex(pd.to_datetime(g.index), freq="Q")
    u = u[~u.index.duplicated(keep="last")].sort_index()
    g = g[~g.index.duplicated(keep="last")].sort_index()
    df = pd.DataFrame({"unemp_rate": u, "gdp_growth_qq": g}).sort_index()
    df["unemp_diff"] = df["unemp_rate"].diff()
    df["gdp_growth_lag1"] = df["gdp_growth_qq"].shift(1)
    df = df.dropna()
    df.index = pd.PeriodIndex(df.index, freq="Q", name="date")
    return df[["unemp_rate", "gdp_growth_qq", "unemp_diff", "gdp_growth_lag1"]]


def _okun_design(panel: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return :math:`(y, X)` for the Beaton Okun regression."""
    y = panel["unemp_diff"].to_numpy(dtype=float)
    X = np.column_stack([
        np.ones(len(panel)),
        panel["gdp_growth_qq"].to_numpy(dtype=float),
        panel["gdp_growth_lag1"].to_numpy(dtype=float),
    ])
    return y, X


# ---------------------------------------------------------------------
# Stock-Watson median-unbiased inversion
# ---------------------------------------------------------------------
def stock_watson_invert(qlrt_statistic: float) -> float:
    """Invert a QLRT statistic to the Stock-Watson :math:`\\lambda`.

    Piecewise-linear interpolation in :data:`STOCK_WATSON_TABLE_3_QLR`.
    """
    lams = np.array(sorted(STOCK_WATSON_TABLE_3_QLR.keys()), dtype=float)
    qs = np.array([STOCK_WATSON_TABLE_3_QLR[int(l)] for l in lams])
    if qlrt_statistic <= qs[0]:
        return 0.0
    if qlrt_statistic >= qs[-1]:
        warnings.warn(
            f"QLRT={qlrt_statistic:.3f} exceeds Stock-Watson table ceiling "
            "(64.016, lambda=30); returning lambda=30.",
            stacklevel=2,
        )
        return 30.0
    return float(np.interp(qlrt_statistic, qs, lams))


# ---------------------------------------------------------------------
# Andrews QLRT (1993)
# ---------------------------------------------------------------------
def _chow_f_break(y: np.ndarray, X: np.ndarray, tau: int,
                  rss_r: float, k: int) -> float:
    """Classical (homoskedastic) Chow F-statistic for a break at row ``tau``.

    The classical form is mandatory: the statistic is inverted against the
    Stock-Watson Table 3 QLR critical values, which assume homoskedasticity.
    """
    T = X.shape[0]
    try:
        b1, *_ = np.linalg.lstsq(X[:tau + 1], y[:tau + 1], rcond=None)
        b2, *_ = np.linalg.lstsq(X[tau + 1:], y[tau + 1:], rcond=None)
    except np.linalg.LinAlgError:
        return np.nan
    rss_u = (float(np.sum((y[:tau + 1] - X[:tau + 1] @ b1) ** 2))
             + float(np.sum((y[tau + 1:] - X[tau + 1:] @ b2) ** 2)))
    if rss_u <= 0 or T - 2 * k <= 0:
        return np.nan
    return ((rss_r - rss_u) / k) / (rss_u / (T - 2 * k))


def andrews_qlrt(panel: pd.DataFrame, trim: float = 0.15) -> Dict[str, Any]:
    """Andrews (1993) Sup-F (Quandt) structural-break test on the Okun regression."""
    y, X = _okun_design(panel)
    T, k = X.shape
    lo = int(np.floor(trim * T))
    hi = int(np.ceil((1 - trim) * T)) - 1
    beta_r, *_ = np.linalg.lstsq(X, y, rcond=None)
    rss_r = float(np.sum((y - X @ beta_r) ** 2))
    best_stat = -np.inf
    best_idx = lo
    for tau in range(lo, hi + 1):
        stat = _chow_f_break(y, X, tau, rss_r, k)
        if np.isfinite(stat) and stat > best_stat:
            best_stat = stat
            best_idx = tau
    return {
        "qlrt_stat": float(best_stat),
        "break_index": int(best_idx),
        "break_date": panel.index[best_idx],
        "trim": float(trim),
        "restricted_rss": rss_r,
        "n_obs": int(T),
    }


# ---------------------------------------------------------------------
# Q-tilde construction (Beaton Eqs 6-7)
# ---------------------------------------------------------------------
def construct_q_tilde(panel: pd.DataFrame, lambda_value: float) -> np.ndarray:
    """Construct the pre-estimated state covariance :math:`\\tilde Q`."""
    y, X = _okun_design(panel)
    T, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    d_weights = (T / (T - k)) * resid ** 2
    omega_tilde = (X.T * d_weights) @ X / T
    return float(lambda_value) ** 2 / T ** 2 * omega_tilde


# ---------------------------------------------------------------------
# Lenza-Primiceri variance scaling
# ---------------------------------------------------------------------
def _lp_scale_vector(period_index: pd.PeriodIndex, s0: float, s1: float,
                     s2: float, rho: float) -> np.ndarray:
    """Build the LP variance scaling vector :math:`s_t` over ``period_index``.

    ``s_t = 1`` everywhere except: 2020Q1 -> ``s0``; 2020Q2 -> ``s1``;
    2020Q3 -> ``s2``; thereafter ``s_t = s2 * rho^(t - 2020Q3)``.
    """
    s = np.ones(len(period_index), dtype=float)
    pivot = pd.Period("2020Q3", freq="Q")
    q1 = pd.Period("2020Q1", freq="Q")
    q2 = pd.Period("2020Q2", freq="Q")
    for i, p in enumerate(period_index):
        if p == q1:
            s[i] = s0
        elif p == q2:
            s[i] = s1
        elif p == pivot:
            s[i] = s2
        elif p > pivot:
            s[i] = s2 * (rho ** (p.ordinal - pivot.ordinal))
    return s


def _kalman_forward(y: np.ndarray, X: np.ndarray, Q: np.ndarray,
                    R_vec: np.ndarray, return_smoother: bool = True
                    ) -> Dict[str, Any]:
    """Forward Kalman filter (optionally storing matrices for the RTS pass).

    State :math:`\\phi_t = \\phi_{t-1} + w_t`, ``w_t ~ N(0, Q)``;
    obs :math:`y_t = X_t \\phi_t + \\varepsilon_t`, ``Var = R_vec[t]``.
    Init :math:`\\phi_0` from OLS on the first min(20, T/3) obs; ``P_0 = 1000 I``.
    """
    T, k = X.shape
    init_n = min(20, max(k + 2, T // 3))
    beta0, *_ = np.linalg.lstsq(X[:init_n], y[:init_n], rcond=None)
    phi = beta0.astype(float)
    P = 1000.0 * np.eye(k)

    phi_pred = np.zeros((T, k))
    P_pred = np.zeros((T, k, k))
    phi_upd = np.zeros((T, k))
    P_upd = np.zeros((T, k, k))
    loglik = 0.0

    for t in range(T):
        phi_p = phi
        P_p = P + Q
        H = X[t]
        v = y[t] - H @ phi_p
        F = float(H @ P_p @ H + R_vec[t])
        if F <= 0:
            return {"loglik": -1e10}
        K = (P_p @ H) / F
        phi = phi_p + K * v
        P = P_p - np.outer(K, H) @ P_p
        P = 0.5 * (P + P.T)
        loglik += -0.5 * (np.log(2 * np.pi * F) + v * v / F)

        if return_smoother:
            phi_pred[t] = phi_p
            P_pred[t] = P_p
            phi_upd[t] = phi
            P_upd[t] = P

    out: Dict[str, Any] = {"loglik": float(loglik)}
    if return_smoother:
        out.update({"phi_pred": phi_pred, "P_pred": P_pred,
                    "phi_upd": phi_upd, "P_upd": P_upd})
    return out


def _kalman_loglik(theta_unconstrained: np.ndarray,
                   y: np.ndarray, X: np.ndarray, Q: np.ndarray,
                   s_template_idx: pd.PeriodIndex,
                   sigma2_init: float) -> float:
    """Negative log-likelihood at unconstrained ``theta``."""
    s0 = 1.0 + np.exp(theta_unconstrained[0])
    s1 = 1.0 + np.exp(theta_unconstrained[1])
    s2 = 1.0 + np.exp(theta_unconstrained[2])
    rho = 1.0 / (1.0 + np.exp(-theta_unconstrained[3]))
    sigma2 = np.exp(theta_unconstrained[4])
    s_vec = _lp_scale_vector(s_template_idx, s0, s1, s2, rho)
    try:
        out = _kalman_forward(y, X, Q, sigma2 * s_vec, return_smoother=False)
    except np.linalg.LinAlgError:
        return 1e10
    return -float(out["loglik"])


def estimate_lp_scaling(panel: pd.DataFrame, q_tilde: np.ndarray,
                        sigma_eps_sq_init: float,
                        n_starts: int = 5) -> Dict[str, Any]:
    """Joint MLE on Lenza-Primiceri scaling + residual variance.

    Maximises the Kalman filter log-likelihood over
    :math:`(\\bar s_0, \\bar s_1, \\bar s_2, \\rho, \\sigma^2_\\varepsilon)`
    with :math:`\\tilde Q` fixed. ``n_starts`` defaults to 5 (reduced from the
    research code's 10) to bound notebook runtime.
    """
    y, X = _okun_design(panel)

    starts: List[Tuple[float, float, float, float]] = [
        (1.0, 10.0, 5.0, 0.3),
        (10.0, 50.0, 25.0, 0.5),
        (50.0, 100.0, 75.0, 0.8),
        (1.0, 100.0, 75.0, 0.3),
        (50.0, 10.0, 5.0, 0.8),
        (10.0, 100.0, 25.0, 0.5),
        (10.0, 50.0, 75.0, 0.3),
        (50.0, 50.0, 25.0, 0.5),
        (1.0, 50.0, 25.0, 0.5),
        (10.0, 10.0, 25.0, 0.8),
    ][:n_starts]

    def to_unconstrained(s0, s1, s2, rho, sigma2):
        return np.array([
            np.log(max(s0 - 1.0, 1e-6)),
            np.log(max(s1 - 1.0, 1e-6)),
            np.log(max(s2 - 1.0, 1e-6)),
            np.log(rho / (1.0 - rho)),
            np.log(max(sigma2, 1e-8)),
        ])

    best: Optional[Dict[str, Any]] = None
    all_lls: List[float] = []
    for (s0, s1, s2, rho) in starts:
        theta0 = to_unconstrained(s0, s1, s2, rho, sigma_eps_sq_init)
        try:
            res = minimize(_kalman_loglik, theta0,
                           args=(y, X, q_tilde, panel.index,
                                 sigma_eps_sq_init),
                           method="Nelder-Mead",
                           options={"xatol": 1e-5, "fatol": 1e-5,
                                    "maxiter": 4000})
        except Exception:  # noqa: BLE001
            all_lls.append(np.nan)
            continue
        ll = -float(res.fun)
        all_lls.append(ll)
        if best is None or ll > best["log_likelihood_best"]:
            best = {
                "s0": float(1.0 + np.exp(res.x[0])),
                "s1": float(1.0 + np.exp(res.x[1])),
                "s2": float(1.0 + np.exp(res.x[2])),
                "rho": float(1.0 / (1.0 + np.exp(-res.x[3]))),
                "sigma_eps_sq": float(np.exp(res.x[4])),
                "log_likelihood_best": ll,
                "converged": bool(res.success),
            }
    if best is None:
        raise RuntimeError("LP MLE failed from every starting point.")
    return {**best, "log_likelihoods_all": all_lls, "starts": starts}


# ---------------------------------------------------------------------
# RTS Kalman smoother (public)
# ---------------------------------------------------------------------
def kalman_filter_smoother(panel: pd.DataFrame, q_tilde: np.ndarray,
                           lp_params: Dict[str, float]) -> Dict[str, Any]:
    """Forward Kalman filter + Rauch-Tung-Striebel backward smoother.

    Returns ``{"phi_smoothed", "phi_smoothed_se", "total_okun_smoothed",
    "total_okun_smoothed_se", "P_smoothed", "time_index", "lp_scaling"}``.
    ``total_okun_smoothed`` is :math:`\\alpha_{1,t} + \\alpha_{2,t}`.
    """
    y, X = _okun_design(panel)
    s_vec = _lp_scale_vector(panel.index, lp_params["s0"], lp_params["s1"],
                             lp_params["s2"], lp_params["rho"])
    R_vec = lp_params["sigma_eps_sq"] * s_vec

    fwd = _kalman_forward(y, X, q_tilde, R_vec, return_smoother=True)
    phi_pred = fwd["phi_pred"]
    P_pred = fwd["P_pred"]
    phi_upd = fwd["phi_upd"]
    P_upd = fwd["P_upd"]
    T, k = X.shape

    phi_sm = np.zeros_like(phi_upd)
    P_sm = np.zeros_like(P_upd)
    phi_sm[-1] = phi_upd[-1]
    P_sm[-1] = P_upd[-1]
    for t in range(T - 2, -1, -1):
        try:
            G = P_upd[t] @ np.linalg.inv(P_pred[t + 1])
        except np.linalg.LinAlgError:
            G = np.zeros((k, k))
        phi_sm[t] = phi_upd[t] + G @ (phi_sm[t + 1] - phi_pred[t + 1])
        P_sm[t] = P_upd[t] + G @ (P_sm[t + 1] - P_pred[t + 1]) @ G.T
        P_sm[t] = 0.5 * (P_sm[t] + P_sm[t].T)

    phi_se = np.sqrt(np.maximum(
        np.stack([np.diag(P_sm[t]) for t in range(T)]), 0.0))
    total = phi_sm[:, 1] + phi_sm[:, 2]
    L = np.array([0.0, 1.0, 1.0])
    total_var = np.array([L @ P_sm[t] @ L for t in range(T)])
    total_se = np.sqrt(np.maximum(total_var, 0.0))

    return {
        "phi_smoothed": phi_sm,
        "phi_smoothed_se": phi_se,
        "total_okun_smoothed": total,
        "total_okun_smoothed_se": total_se,
        "P_smoothed": P_sm,
        "time_index": panel.index,
        "lp_scaling": s_vec,
    }


# ---------------------------------------------------------------------
# OLS baseline + regime averages
# ---------------------------------------------------------------------
def ols_baseline(panel: pd.DataFrame, hac_lags: int = 4) -> Dict[str, Any]:
    """Static OLS Beaton regression with Newey-West HAC SEs."""
    y, X = _okun_design(panel)
    if sm is None:
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        return {"alpha0": float(beta[0]), "alpha1": float(beta[1]),
                "alpha2": float(beta[2]),
                "total_okun": float(beta[1] + beta[2]),
                "total_okun_se": float("nan"), "n_obs": int(len(y))}
    model = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})
    coef = model.params
    se = model.bse
    cov = model.cov_params()
    var_total = float(cov[1, 1] + cov[2, 2] + 2 * cov[1, 2])
    return {
        "alpha0": float(coef[0]), "alpha0_se": float(se[0]),
        "alpha1": float(coef[1]), "alpha1_se": float(se[1]),
        "alpha2": float(coef[2]), "alpha2_se": float(se[2]),
        "total_okun": float(coef[1] + coef[2]),
        "total_okun_se": float(np.sqrt(max(var_total, 0.0))),
        "n_obs": int(model.nobs),
    }


def regime_averages(smoother: Dict[str, Any]) -> pd.DataFrame:
    """Average the smoothed total-Okun coefficient over :data:`REGIME_WINDOWS`."""
    idx = smoother["time_index"]
    total = smoother["total_okun_smoothed"]
    out_rows = []
    for name, (start, end) in REGIME_WINDOWS.items():
        mask = (idx >= start) & (idx <= end)
        if mask.sum() == 0:
            continue
        vals = total[mask]
        sem = (float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
               if len(vals) > 1 else float("nan"))
        out_rows.append((name, float(np.mean(vals)), sem,
                         int(mask.sum()), str(start), str(end)))
    return pd.DataFrame(out_rows,
                        columns=["regime", "mean", "sem", "n",
                                 "start", "end"]).set_index("regime")


# ---------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------
def estimate_okun_tvp(panel: pd.DataFrame,
                      drop_covid_for_qlrt: bool = True,
                      n_starts: int = 5) -> Dict[str, Any]:
    """Run the full Beaton-SW-LP pipeline on a pre-built Okun panel.

    Parameters
    ----------
    panel : pandas.DataFrame
        Output of :func:`build_panel`.
    drop_covid_for_qlrt : bool, default True
        Compute the Andrews QLRT (and :math:`\\lambda`) on the pre-COVID
        sub-sample (<= 2019Q4) so the 2020 outliers feed the LP scaling, not
        the persistent state drift.
    n_starts : int, default 5
        MLE restarts for the LP scaling.

    Returns
    -------
    dict
        ``{"panel", "ols_full", "qlrt", "lambda", "q_tilde", "lp",
        "smoother", "regimes"}``.
    """
    ols_full = ols_baseline(panel)

    if drop_covid_for_qlrt:
        qlrt_panel = panel.loc[panel.index <= pd.Period("2019Q4")]
        if len(qlrt_panel) < 12:
            qlrt_panel = panel
    else:
        qlrt_panel = panel
    qlrt = andrews_qlrt(qlrt_panel)
    lam = stock_watson_invert(qlrt["qlrt_stat"])
    q_tilde = construct_q_tilde(qlrt_panel, lam)

    y_pre, X_pre = _okun_design(qlrt_panel)
    b_pre, *_ = np.linalg.lstsq(X_pre, y_pre, rcond=None)
    resid_pre = y_pre - X_pre @ b_pre
    sigma2_init = float(np.var(resid_pre, ddof=X_pre.shape[1]))

    lp = estimate_lp_scaling(panel, q_tilde, sigma2_init, n_starts=n_starts)
    smoother = kalman_filter_smoother(panel, q_tilde, lp)
    regimes = regime_averages(smoother)

    return {
        "panel": panel,
        "ols_full": ols_full,
        "qlrt": qlrt,
        "lambda": float(lam),
        "q_tilde": q_tilde,
        "sigma2_init": sigma2_init,
        "lp": lp,
        "smoother": smoother,
        "regimes": regimes,
    }
