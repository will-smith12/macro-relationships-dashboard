#!/usr/bin/env python3

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "fred_train.csv"
TEST_PATH = BASE_DIR / "fred_test.csv"
OUTPUT_PATH = BASE_DIR / "vecm_forecast_final.png"
ERROR_BY_REGIME_PATH = BASE_DIR / "forecast_errors_by_regime.png"
HORIZON_OUTPUT_PATH = BASE_DIR / "forecast_by_horizon.png"

REQUIRED_COLUMNS = ["date", "CPI", "FPP"]

BETA_FPP = 1.099
TRAINING_ECT_MEAN = 0.072949
GAMMA_THRESHOLD = -0.4798
VECM_LAG_LENGTH = 2
CONFIDENCE_LEVEL_Z = 1.96
FORECAST_HORIZONS = [1, 3, 6, 12, 24]


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["date"])
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"{path.name} is missing required columns: {missing_columns}")

    data = data[REQUIRED_COLUMNS].sort_values("date").reset_index(drop=True)
    if data["date"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate dates")
    if (data[["CPI", "FPP"]] <= 0).any().any():
        raise ValueError("CPI and FPP must be positive to compute logs")

    data["log_CPI"] = np.log(data["CPI"])
    data["log_FPP"] = np.log(data["FPP"])
    data["ECT"] = data["log_CPI"] - BETA_FPP * data["log_FPP"] - TRAINING_ECT_MEAN
    return data


def build_vecm_design(log_values: np.ndarray, ect: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Model: Δy_t = c + α ECT_{t-1} + Γ1 Δy_{t-1} + Γ2 Δy_{t-2} + ε_t
    diffs = np.diff(log_values, axis=0)
    x_rows = []
    y_rows = []

    for t in range(VECM_LAG_LENGTH + 1, len(log_values)):
        x_rows.append(
            [
                1.0,
                ect[t - 1],
                diffs[t - 2, 0],
                diffs[t - 2, 1],
                diffs[t - 3, 0],
                diffs[t - 3, 1],
            ]
        )
        y_rows.append(diffs[t - 1])

    return np.asarray(x_rows), np.asarray(y_rows)


def matrix_product(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(left[:, :, None] * right[None, :, :], axis=1)


def estimate_training_vecm(train: pd.DataFrame) -> tuple[np.ndarray, float]:
    log_values = train[["log_CPI", "log_FPP"]].to_numpy()
    x_train, y_train = build_vecm_design(log_values, train["ECT"].to_numpy())

    coefficients, _, _, _ = np.linalg.lstsq(x_train, y_train, rcond=None)
    residuals = y_train - matrix_product(x_train, coefficients)
    cpi_residual_std = residuals[:, 0].std(ddof=x_train.shape[1])

    return coefficients, cpi_residual_std


def estimate_training_ar2(train: pd.DataFrame) -> np.ndarray:
    # Model: Δlog(CPI)_t = c + phi1 * Δlog(CPI)_{t-1} + phi2 * Δlog(CPI)_{t-2} + ε_t
    delta_log_cpi = np.diff(train["log_CPI"].to_numpy())
    x_rows = []
    y_rows = []

    for t in range(2, len(delta_log_cpi)):
        x_rows.append([1.0, delta_log_cpi[t - 1], delta_log_cpi[t - 2]])
        y_rows.append(delta_log_cpi[t])

    coefficients, _, _, _ = np.linalg.lstsq(
        np.asarray(x_rows),
        np.asarray(y_rows),
        rcond=None,
    )
    return coefficients


def vecm_step(log_history: list[np.ndarray], coefficients: np.ndarray) -> np.ndarray:
    last = log_history[-1]
    diff_lag_1 = log_history[-1] - log_history[-2]
    diff_lag_2 = log_history[-2] - log_history[-3]
    ect_last = last[0] - BETA_FPP * last[1] - TRAINING_ECT_MEAN
    x_t = np.asarray(
        [
            1.0,
            ect_last,
            diff_lag_1[0],
            diff_lag_1[1],
            diff_lag_2[0],
            diff_lag_2[1],
        ]
    )
    return last + x_t @ coefficients


def ar2_step(log_cpi_history: list[float], ar2_coefficients: np.ndarray) -> float:
    diff_lag_1 = log_cpi_history[-1] - log_cpi_history[-2]
    diff_lag_2 = log_cpi_history[-2] - log_cpi_history[-3]
    next_delta = float(np.dot([1.0, diff_lag_1, diff_lag_2], ar2_coefficients))
    return log_cpi_history[-1] + next_delta


def make_forecasts(
    train: pd.DataFrame,
    test: pd.DataFrame,
    coefficients: np.ndarray,
    cpi_residual_std: float,
    ar2_coefficients: np.ndarray,
) -> pd.DataFrame:
    combined = pd.concat([train, test], ignore_index=True)
    log_values = combined[["log_CPI", "log_FPP"]].to_numpy()
    ect = combined["ECT"].to_numpy()
    diffs = np.diff(log_values, axis=0)

    train_size = len(train)
    rows = []

    for t in range(train_size, len(combined)):
        x_t = np.asarray(
            [
                1.0,
                ect[t - 1],
                diffs[t - 2, 0],
                diffs[t - 2, 1],
                diffs[t - 3, 0],
                diffs[t - 3, 1],
            ]
        )

        forecast_delta_log_cpi = float(np.dot(x_t, coefficients[:, 0]))
        forecast_log_cpi = log_values[t - 1, 0] + forecast_delta_log_cpi
        lower_log = forecast_log_cpi - CONFIDENCE_LEVEL_Z * cpi_residual_std
        upper_log = forecast_log_cpi + CONFIDENCE_LEVEL_Z * cpi_residual_std
        ar2_delta_log_cpi = float(
            np.dot(
                [1.0, diffs[t - 2, 0], diffs[t - 3, 0]],
                ar2_coefficients,
            )
        )
        ar2_forecast_log_cpi = log_values[t - 1, 0] + ar2_delta_log_cpi

        previous_cpi = combined.loc[t - 1, "CPI"]
        rows.append(
            {
                "date": combined.loc[t, "date"],
                "actual_cpi": combined.loc[t, "CPI"],
                "previous_cpi": previous_cpi,
                "ECT": combined.loc[t, "ECT"],
                "regime": "Turbulent"
                if combined.loc[t, "ECT"] <= GAMMA_THRESHOLD
                else "Normal",
                "vecm_forecast": np.exp(forecast_log_cpi),
                "vecm_lower_95": np.exp(lower_log),
                "vecm_upper_95": np.exp(upper_log),
                "ar2_forecast": np.exp(ar2_forecast_log_cpi),
                "naive_forecast": previous_cpi,
            }
        )

    return pd.DataFrame(rows)


def make_horizon_forecasts(
    train: pd.DataFrame,
    test: pd.DataFrame,
    vecm_coefficients: np.ndarray,
    ar2_coefficients: np.ndarray,
    horizons: list[int],
) -> pd.DataFrame:
    combined = pd.concat([train, test], ignore_index=True).reset_index(drop=True)
    log_values = combined[["log_CPI", "log_FPP"]].to_numpy()
    train_size = len(train)
    n_test = len(test)
    rows = []

    for horizon in horizons:
        if n_test <= horizon:
            continue

        for origin_offset in range(0, n_test - horizon):
            origin_index = train_size + origin_offset
            target_index = origin_index + horizon

            vecm_history = [log_values[i].copy() for i in range(origin_index - 2, origin_index + 1)]
            for _ in range(horizon):
                vecm_history.append(vecm_step(vecm_history, vecm_coefficients))
            vecm_forecast_log_cpi = vecm_history[-1][0]

            ar2_history = [
                float(combined.loc[i, "log_CPI"])
                for i in range(origin_index - 2, origin_index + 1)
            ]
            for _ in range(horizon):
                ar2_history.append(ar2_step(ar2_history, ar2_coefficients))
            ar2_forecast_log_cpi = ar2_history[-1]

            rows.append(
                {
                    "horizon": horizon,
                    "origin_date": combined.loc[origin_index, "date"],
                    "target_date": combined.loc[target_index, "date"],
                    "actual_cpi": combined.loc[target_index, "CPI"],
                    "vecm_forecast": np.exp(vecm_forecast_log_cpi),
                    "ar2_forecast": np.exp(ar2_forecast_log_cpi),
                }
            )

    return pd.DataFrame(rows)


def direction_accuracy(actual: pd.Series, forecast: pd.Series, previous: pd.Series) -> float:
    actual_direction = np.sign(actual.to_numpy() - previous.to_numpy())
    forecast_direction = np.sign(forecast.to_numpy() - previous.to_numpy())
    return float((actual_direction == forecast_direction).mean() * 100)


def compute_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    actual = forecasts["actual_cpi"]
    previous = forecasts["previous_cpi"]

    metric_rows = []
    for model_name, forecast_column in [
        ("VECM", "vecm_forecast"),
        ("AR(2)", "ar2_forecast"),
        ("Naive", "naive_forecast"),
    ]:
        forecast = forecasts[forecast_column]
        error = forecast - actual
        metric_rows.append(
            {
                "model": model_name,
                "RMSE": np.sqrt(np.mean(error**2)),
                "MAE": np.mean(np.abs(error)),
                "MAPE": np.mean(np.abs(error / actual)) * 100,
                "Direction accuracy (%)": direction_accuracy(actual, forecast, previous),
            }
        )

    metrics = pd.DataFrame(metric_rows).set_index("model")

    comparison = pd.DataFrame(
        {
            "VECM": metrics.loc["VECM"],
            "AR(2)": metrics.loc["AR(2)"],
            "Naive": metrics.loc["Naive"],
            "VECM beats naive": [
                metrics.loc["VECM", "RMSE"] < metrics.loc["Naive", "RMSE"],
                metrics.loc["VECM", "MAE"] < metrics.loc["Naive", "MAE"],
                metrics.loc["VECM", "MAPE"] < metrics.loc["Naive", "MAPE"],
                metrics.loc["VECM", "Direction accuracy (%)"]
                > metrics.loc["Naive", "Direction accuracy (%)"],
            ],
            "AR(2) beats naive": [
                metrics.loc["AR(2)", "RMSE"] < metrics.loc["Naive", "RMSE"],
                metrics.loc["AR(2)", "MAE"] < metrics.loc["Naive", "MAE"],
                metrics.loc["AR(2)", "MAPE"] < metrics.loc["Naive", "MAPE"],
                metrics.loc["AR(2)", "Direction accuracy (%)"]
                > metrics.loc["Naive", "Direction accuracy (%)"],
            ],
        },
        index=["RMSE", "MAE", "MAPE", "Direction accuracy (%)"],
    )
    return comparison


def model_metric_values(
    forecasts: pd.DataFrame,
    forecast_column: str,
) -> dict[str, float]:
    actual = forecasts["actual_cpi"]
    forecast = forecasts[forecast_column]
    error = forecast - actual

    return {
        "RMSE": np.sqrt(np.mean(error**2)),
        "MAE": np.mean(np.abs(error)),
        "MAPE": np.mean(np.abs(error / actual)) * 100,
        "Direction": direction_accuracy(actual, forecast, forecasts["previous_cpi"]),
    }


def compute_regime_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for regime in ["Turbulent", "Normal"]:
        regime_forecasts = forecasts[forecasts["regime"] == regime]
        for model_name, forecast_column in [
            ("VECM", "vecm_forecast"),
            ("AR(2)", "ar2_forecast"),
        ]:
            metrics = model_metric_values(regime_forecasts, forecast_column)
            rows.append({"Regime": regime, "Model": model_name, **metrics})

    return pd.DataFrame(rows)


def print_turbulent_months(forecasts: pd.DataFrame) -> None:
    turbulent = forecasts.loc[forecasts["regime"] == "Turbulent", ["date", "ECT"]].copy()
    turbulent["date"] = turbulent["date"].dt.strftime("%Y-%m")
    turbulent["ECT"] = turbulent["ECT"].round(6)

    print("Turbulent months in the test period")
    print("-----------------------------------")
    if turbulent.empty:
        print("None")
    else:
        print(turbulent.to_string(index=False))
    print()


def longest_run(labels: np.ndarray) -> int:
    max_run = 0
    current_run = 0

    for label in labels:
        if label == 1:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    return max_run


def clustering_permutation_test(forecasts: pd.DataFrame, n_permutations: int = 10_000) -> None:
    rng = np.random.default_rng(12345)
    labels = (forecasts["ECT"].to_numpy() <= GAMMA_THRESHOLD).astype(int)
    actual_longest_run = longest_run(labels)
    shuffled_longest_runs = np.empty(n_permutations, dtype=int)

    for i in range(n_permutations):
        shuffled_labels = rng.permutation(labels)
        shuffled_longest_runs[i] = longest_run(shuffled_labels)

    p95 = np.percentile(shuffled_longest_runs, 95)
    empirical_p_value = (shuffled_longest_runs >= actual_longest_run).mean()
    is_unusual = empirical_p_value < 0.05

    results = pd.DataFrame(
        [
            {
                "actual_longest_run": actual_longest_run,
                "shuffle_mean_longest_run": shuffled_longest_runs.mean(),
                "shuffle_95th_percentile": p95,
                "empirical_p_value": empirical_p_value,
                "unusual_at_5pct": "Yes" if is_unusual else "No",
            }
        ]
    )

    print("TEST 1: Clustering permutation test for turbulent months")
    print("--------------------------------------------------------")
    print(
        results.to_string(
            index=False,
            formatters={
                "shuffle_mean_longest_run": "{:.4f}".format,
                "shuffle_95th_percentile": "{:.4f}".format,
                "empirical_p_value": "{:.4f}".format,
            },
        )
    )
    if is_unusual:
        print(
            f"A consecutive block of {actual_longest_run} turbulent months is statistically unusual "
            "under random ordering at the 5% level."
        )
    else:
        print(
            f"A consecutive block of {actual_longest_run} turbulent months is not statistically unusual "
            "under random ordering at the 5% level."
        )
    print()


def diebold_mariano_test(
    forecasts: pd.DataFrame,
    sample_name: str,
) -> dict[str, object]:
    vecm_error = forecasts["vecm_forecast"] - forecasts["actual_cpi"]
    ar2_error = forecasts["ar2_forecast"] - forecasts["actual_cpi"]
    loss_diff = vecm_error.pow(2) - ar2_error.pow(2)
    n_obs = len(loss_diff)

    if n_obs < 2:
        return {
            "Sample": sample_name,
            "n": n_obs,
            "mean_loss_diff": np.nan,
            "DM_stat": np.nan,
            "p_value": np.nan,
            "Conclusion": "Insufficient observations",
        }

    loss_diff_mean = loss_diff.mean()
    loss_diff_var = loss_diff.var(ddof=1)
    if loss_diff_var <= 0 or not np.isfinite(loss_diff_var):
        dm_hln = np.nan
        p_value = np.nan
    else:
        # h = 1, so the long-run variance is the variance of d_t. The
        # Harvey-Leybourne-Newbold correction is sqrt((T - 1) / T).
        dm_stat = loss_diff_mean / np.sqrt(loss_diff_var / n_obs)
        hln_correction = np.sqrt((n_obs - 1) / n_obs)
        dm_hln = dm_stat * hln_correction
        p_value = 2 * stats.t.sf(abs(dm_hln), df=n_obs - 1)

    if np.isfinite(p_value) and p_value < 0.05 and loss_diff_mean < 0:
        conclusion = "VECM significantly more accurate than AR(2)"
    elif np.isfinite(p_value) and p_value < 0.05 and loss_diff_mean > 0:
        conclusion = "AR(2) significantly more accurate than VECM"
    else:
        conclusion = "Difference not statistically significant"

    if sample_name == "Turbulent":
        conclusion += " (only 6 obs; very low power)"

    return {
        "Sample": sample_name,
        "n": n_obs,
        "mean_loss_diff": loss_diff_mean,
        "DM_stat": dm_hln,
        "p_value": p_value,
        "Conclusion": conclusion,
    }


def print_diebold_mariano_tests(forecasts: pd.DataFrame) -> None:
    samples = [
        ("Full test period", forecasts),
        ("Turbulent", forecasts[forecasts["regime"] == "Turbulent"]),
        ("Normal", forecasts[forecasts["regime"] == "Normal"]),
    ]
    results = pd.DataFrame(
        diebold_mariano_test(sample_forecasts, sample_name)
        for sample_name, sample_forecasts in samples
    )

    print("TEST 2: Diebold-Mariano tests, VECM vs AR(2)")
    print("--------------------------------------------")
    print(
        results.to_string(
            index=False,
            formatters={
                "mean_loss_diff": "{:.6f}".format,
                "DM_stat": "{:.4f}".format,
                "p_value": "{:.4f}".format,
            },
        )
    )
    print("Loss differential is d_t = VECM squared error - AR(2) squared error.")
    print("Negative mean_loss_diff means VECM has lower average squared error.")
    print()


def h_step_dm_test(loss_diff: pd.Series, horizon: int) -> tuple[float, float]:
    values = loss_diff.to_numpy(dtype=float)
    n_obs = len(values)
    centered = values - values.mean()

    if n_obs < 2:
        return np.nan, np.nan

    gamma_0 = float(np.mean(centered * centered))
    long_run_variance = gamma_0
    max_lag = min(horizon - 1, n_obs - 1)

    for lag in range(1, max_lag + 1):
        autocov = float(np.mean(centered[lag:] * centered[:-lag]))
        bartlett_weight = 1.0 - lag / (max_lag + 1.0)
        long_run_variance += 2.0 * bartlett_weight * autocov

    if long_run_variance <= 0 or not np.isfinite(long_run_variance):
        return np.nan, np.nan

    dm_stat = values.mean() / np.sqrt(long_run_variance / n_obs)
    hln_factor = (n_obs + 1 - 2 * horizon + horizon * (horizon - 1) / n_obs) / n_obs
    if hln_factor <= 0:
        return np.nan, np.nan

    dm_hln = dm_stat * np.sqrt(hln_factor)
    p_value = 2 * stats.t.sf(abs(dm_hln), df=n_obs - 1)
    return dm_hln, p_value


def compute_horizon_metrics(horizon_forecasts: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for horizon, group in horizon_forecasts.groupby("horizon", sort=True):
        actual = group["actual_cpi"]
        vecm_error = group["vecm_forecast"] - actual
        ar2_error = group["ar2_forecast"] - actual
        loss_diff = vecm_error.pow(2) - ar2_error.pow(2)
        dm_stat, dm_p_value = h_step_dm_test(loss_diff, int(horizon))

        vecm_rmse = np.sqrt(np.mean(vecm_error**2))
        ar2_rmse = np.sqrt(np.mean(ar2_error**2))
        rows.append(
            {
                "horizon": int(horizon),
                "n_pairs": len(group),
                "VECM_RMSE": vecm_rmse,
                "AR2_RMSE": ar2_rmse,
                "VECM_RMSE_improvement_pct": (ar2_rmse - vecm_rmse) / ar2_rmse * 100,
                "VECM_MAE": np.mean(np.abs(vecm_error)),
                "AR2_MAE": np.mean(np.abs(ar2_error)),
                "VECM_MAPE": np.mean(np.abs(vecm_error / actual)) * 100,
                "AR2_MAPE": np.mean(np.abs(ar2_error / actual)) * 100,
                "DM_stat": dm_stat,
                "DM_p_value": dm_p_value,
            }
        )

    return pd.DataFrame(rows)


def print_horizon_results(horizon_metrics: pd.DataFrame) -> None:
    print("Multi-horizon rolling-origin forecast comparison")
    print("------------------------------------------------")
    for row in horizon_metrics.itertuples(index=False):
        print(
            f"Horizon h={row.horizon}: based on {row.n_pairs} forecast-actual pairs "
            f"(n_test - h)."
        )
    print()

    summary = horizon_metrics[
        [
            "horizon",
            "VECM_RMSE",
            "AR2_RMSE",
            "VECM_RMSE_improvement_pct",
            "DM_p_value",
        ]
    ].copy()
    print(
        summary.to_string(
            index=False,
            formatters={
                "VECM_RMSE": "{:.4f}".format,
                "AR2_RMSE": "{:.4f}".format,
                "VECM_RMSE_improvement_pct": "{:.2f}".format,
                "DM_p_value": "{:.4f}".format,
            },
        )
    )
    print()


def plot_horizon_rmse(horizon_metrics: pd.DataFrame) -> None:
    if "seaborn-v0_8-whitegrid" in plt.style.available:
        plt.style.use("seaborn-v0_8-whitegrid")
    else:
        plt.style.use("default")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(
        horizon_metrics["horizon"],
        horizon_metrics["VECM_RMSE"],
        color="red",
        marker="o",
        linewidth=2.2,
        label="VECM",
    )
    ax.plot(
        horizon_metrics["horizon"],
        horizon_metrics["AR2_RMSE"],
        color="grey",
        marker="o",
        linewidth=2.2,
        label="AR(2)",
    )

    ax.set_title("VECM vs AR(2) Forecast Accuracy by Horizon", fontsize=15, pad=12)
    ax.set_xlabel("Forecast horizon (months)")
    ax.set_ylabel("RMSE")
    ax.set_xticks(FORECAST_HORIZONS)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(HORIZON_OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_forecasts(forecasts: pd.DataFrame) -> None:
    if "seaborn-v0_8-whitegrid" in plt.style.available:
        plt.style.use("seaborn-v0_8-whitegrid")
    else:
        plt.style.use("default")

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(
        forecasts["date"],
        forecasts["actual_cpi"],
        color="royalblue",
        linewidth=2.2,
        label="Actual CPI",
    )
    ax.plot(
        forecasts["date"],
        forecasts["vecm_forecast"],
        color="red",
        linestyle="--",
        linewidth=2.0,
        label="VECM forecast",
    )
    ax.plot(
        forecasts["date"],
        forecasts["naive_forecast"],
        color="grey",
        linestyle=":",
        linewidth=2.0,
        label="Naive forecast",
    )
    ax.plot(
        forecasts["date"],
        forecasts["ar2_forecast"],
        color="darkorange",
        linestyle="-.",
        linewidth=2.0,
        label="AR(2)",
    )
    ax.fill_between(
        forecasts["date"],
        forecasts["vecm_lower_95"],
        forecasts["vecm_upper_95"],
        color="red",
        alpha=0.12,
        label="95% confidence band",
    )

    ax.set_title("VECM Out-of-Sample CPI Forecast: Jun 2022 - present", fontsize=15, pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("CPI level")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_forecast_errors_by_regime(forecasts: pd.DataFrame) -> None:
    if "seaborn-v0_8-whitegrid" in plt.style.available:
        plt.style.use("seaborn-v0_8-whitegrid")
    else:
        plt.style.use("default")

    error_data = forecasts.copy()
    error_data["VECM error"] = error_data["vecm_forecast"] - error_data["actual_cpi"]
    error_data["AR(2) error"] = error_data["ar2_forecast"] - error_data["actual_cpi"]

    fig, ax = plt.subplots(figsize=(13, 6))
    for row in error_data.itertuples(index=False):
        if row.regime == "Turbulent":
            ax.axvspan(
                row.date,
                row.date + pd.DateOffset(months=1),
                color="red",
                alpha=0.14,
                linewidth=0,
            )

    ax.plot(
        error_data["date"],
        error_data["VECM error"],
        color="red",
        linewidth=2.0,
        label="VECM forecast error",
    )
    ax.plot(
        error_data["date"],
        error_data["AR(2) error"],
        color="darkorange",
        linewidth=2.0,
        linestyle="-.",
        label="AR(2) forecast error",
    )
    ax.axhline(0, color="black", linestyle="--", linewidth=1.2)

    ax.set_title("Forecast Errors by Regime: Test Period", fontsize=15, pad=12)
    ax.set_xlabel("Date")
    ax.set_ylabel("Forecast error (forecast - actual CPI)")
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(ERROR_BY_REGIME_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    train = load_data(TRAIN_PATH)
    test = load_data(TEST_PATH)

    coefficients, cpi_residual_std = estimate_training_vecm(train)
    ar2_coefficients = estimate_training_ar2(train)
    forecasts = make_forecasts(train, test, coefficients, cpi_residual_std, ar2_coefficients)
    horizon_forecasts = make_horizon_forecasts(
        train,
        test,
        coefficients,
        ar2_coefficients,
        FORECAST_HORIZONS,
    )
    horizon_metrics = compute_horizon_metrics(horizon_forecasts)
    metrics = compute_metrics(forecasts)

    print("Linear VECM setup")
    print("-----------------")
    print(f"Training sample: {train['date'].min():%Y-%m-%d} to {train['date'].max():%Y-%m-%d}")
    print(f"Test sample: {test['date'].min():%Y-%m-%d} to {test['date'].max():%Y-%m-%d}")
    print(f"Cointegrating vector: ECT = log(CPI) - {BETA_FPP} * log(FPP)")
    print(f"Demeaning constant: training mean = {TRAINING_ECT_MEAN}")
    print(f"CPI equation residual std. error in log units: {cpi_residual_std:.6f}")
    print(
        "AR(2) on d.log(CPI): "
        f"const={ar2_coefficients[0]:.6f}, "
        f"lag1={ar2_coefficients[1]:.6f}, "
        f"lag2={ar2_coefficients[2]:.6f}"
    )
    print()

    print("Out-of-sample forecast accuracy")
    print("-------------------------------")
    printable_metrics = metrics.copy()
    printable_metrics["VECM"] = printable_metrics["VECM"].map(lambda value: f"{value:.4f}")
    printable_metrics["AR(2)"] = printable_metrics["AR(2)"].map(lambda value: f"{value:.4f}")
    printable_metrics["Naive"] = printable_metrics["Naive"].map(lambda value: f"{value:.4f}")
    printable_metrics["VECM beats naive"] = printable_metrics["VECM beats naive"].map(
        lambda value: "Yes" if value else "No"
    )
    printable_metrics["AR(2) beats naive"] = printable_metrics["AR(2) beats naive"].map(
        lambda value: "Yes" if value else "No"
    )
    print(printable_metrics.to_string())
    print()
    print_horizon_results(horizon_metrics)

    plot_forecasts(forecasts)
    regime_metrics = compute_regime_metrics(forecasts)

    print()
    print("Forecast accuracy by test-period regime")
    print("---------------------------------------")
    printable_regime_metrics = regime_metrics.copy()
    for column in ["RMSE", "MAE", "MAPE", "Direction"]:
        printable_regime_metrics[column] = printable_regime_metrics[column].map(
            lambda value: f"{value:.4f}"
        )
    print(printable_regime_metrics.to_string(index=False))

    print()
    print_turbulent_months(forecasts)
    clustering_permutation_test(forecasts)
    print_diebold_mariano_tests(forecasts)

    plot_forecast_errors_by_regime(forecasts)
    plot_horizon_rmse(horizon_metrics)
    print(f"\nChart saved to {OUTPUT_PATH}")
    print(f"Forecast error chart saved to {ERROR_BY_REGIME_PATH}")
    print(f"Horizon RMSE chart saved to {HORIZON_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
