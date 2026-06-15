from pathlib import Path
from typing import Optional

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "fred_data_extended.csv"
OUTPUT_PATH = BASE_DIR / "regime_detection_final.png"

REQUIRED_COLUMNS = ["date", "CPI", "CPI_core", "FPP"]
TRAIN_START = pd.Timestamp("1957-01-01")
TRAIN_END = pd.Timestamp("2022-05-01")
TEST_START = pd.Timestamp("2022-06-01")

BETA_FPP = 1.099
GAMMA_THRESHOLD = -0.4798
TRAINING_ECT_MEAN = 0.072949


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["date"])
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"{path.name} is missing required columns: {missing_columns}")

    data = data[REQUIRED_COLUMNS].sort_values("date").reset_index(drop=True)
    if data["date"].duplicated().any():
        raise ValueError(f"{path.name} contains duplicate dates")
    if (data[["CPI", "FPP"]] <= 0).any().any():
        raise ValueError("CPI and FPP must be positive to compute log values")

    return data


def add_raw_ect(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["ECT_raw"] = np.log(data["CPI"]) - BETA_FPP * np.log(data["FPP"])
    return data


def split_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = data[(data["date"] >= TRAIN_START) & (data["date"] <= TRAIN_END)].copy()
    test = data[data["date"] >= TEST_START].copy()

    if train.empty:
        raise ValueError("Training set is empty")
    if test.empty:
        raise ValueError("Test set is empty")

    return train, test


def add_demeaned_ect_and_regimes(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["ECT"] = data["ECT_raw"] - TRAINING_ECT_MEAN
    data["regime"] = np.where(data["ECT"] <= GAMMA_THRESHOLD, "Turbulent", "Normal")
    return data


def print_ect_stats(train: pd.DataFrame, test: pd.DataFrame) -> None:
    summary = pd.DataFrame(
        {
            "min": [train["ECT"].min(), test["ECT"].min()],
            "max": [train["ECT"].max(), test["ECT"].max()],
            "mean": [train["ECT"].mean(), test["ECT"].mean()],
        },
        index=["Training demeaned ECT", "Test demeaned ECT"],
    )

    print("Demeaned ECT summary statistics")
    print(summary.round(6).to_string())
    print()


def print_monthly_classification(test: pd.DataFrame) -> None:
    monthly = test[["date", "regime"]].copy()
    monthly["month"] = monthly["date"].dt.strftime("%Y-%m")

    print("Month-by-month regime classification for the test period")
    print(monthly[["month", "regime"]].to_string(index=False))
    print()


def print_regime_summary(train: pd.DataFrame, test: pd.DataFrame) -> None:
    train_counts = train["regime"].value_counts().reindex(["Turbulent", "Normal"], fill_value=0)
    counts = test["regime"].value_counts().reindex(["Turbulent", "Normal"], fill_value=0)
    summary = pd.DataFrame(
        {
            "test_months": counts,
            "test_pct": (counts / len(test) * 100).round(1),
            "training_months": train_counts,
            "training_pct": (train_counts / len(train) * 100).round(1),
        }
    )
    summary["pct_point_difference_vs_training"] = (
        summary["test_pct"] - summary["training_pct"]
    ).round(1)

    print("Regime summary using demeaned ECT")
    print(summary.to_string())
    print()


def annotate_event(
    ax: plt.Axes,
    date: pd.Timestamp,
    label: str,
    y_value: float,
    y_offset: float,
    *,
    x_anchor: Optional[pd.Timestamp] = None,
) -> None:
    annotation_date = x_anchor if x_anchor is not None else date
    ax.annotate(
        label,
        xy=(annotation_date, y_value),
        xytext=(annotation_date, y_value + y_offset),
        ha="center",
        va="bottom" if y_offset >= 0 else "top",
        fontsize=10,
        arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 1.0},
        color="#333333",
    )


def nearest_ect(data: pd.DataFrame, date: pd.Timestamp) -> float:
    return data.loc[data["date"].sub(date).abs().idxmin(), "ECT"]


def plot_regime_chart(data: pd.DataFrame, output_path: Path) -> None:
    if "seaborn-v0_8-whitegrid" in plt.style.available:
        plt.style.use("seaborn-v0_8-whitegrid")
    else:
        plt.style.use("default")

    colors = {
        "Turbulent": "#e74c3c",
        "Normal": "#9ecae1",
    }

    fig, ax = plt.subplots(figsize=(16, 6.5))
    for row in data.itertuples(index=False):
        ax.axvspan(
            row.date,
            row.date + pd.DateOffset(months=1),
            color=colors[row.regime],
            alpha=0.18,
            linewidth=0,
        )

    ax.plot(
        data["date"],
        data["ECT"],
        color="#1f2937",
        linewidth=1.7,
        label="Demeaned ECT",
    )
    ax.axhline(
        GAMMA_THRESHOLD,
        color="red",
        linestyle="--",
        linewidth=1.6,
        label=f"Threshold gamma = {GAMMA_THRESHOLD:.4f}",
    )
    ax.axhline(
        0,
        color="grey",
        linestyle="--",
        linewidth=1.4,
        label="Zero / equilibrium",
    )

    out_of_sample_start = pd.Timestamp("2022-06-01")
    ax.axvline(
        out_of_sample_start,
        color="black",
        linestyle="--",
        linewidth=1.4,
    )

    y_min = min(data["ECT"].min(), GAMMA_THRESHOLD, 0)
    y_max = max(data["ECT"].max(), GAMMA_THRESHOLD, 0)
    y_range = y_max - y_min if y_max > y_min else 0.01
    ax.set_ylim(y_min - 0.18 * y_range, y_max + 0.28 * y_range)

    ax.text(
        out_of_sample_start + pd.DateOffset(months=3),
        y_max + 0.18 * y_range,
        "Out-of-sample period starts",
        ha="left",
        va="center",
        fontsize=10,
        color="black",
    )

    annotate_event(
        ax,
        pd.Timestamp("1973-10-01"),
        "Oil embargo",
        nearest_ect(data, pd.Timestamp("1973-10-01")),
        0.18 * y_range,
    )
    annotate_event(
        ax,
        pd.Timestamp("1979-01-01"),
        "Iranian Revolution",
        nearest_ect(data, pd.Timestamp("1979-01-01")),
        -0.20 * y_range,
    )
    annotate_event(
        ax,
        pd.Timestamp("2022-06-01"),
        "Russia-Ukraine peak",
        nearest_ect(data, pd.Timestamp("2022-06-01")),
        0.16 * y_range,
    )

    ax.set_title("TVECM Regime Classification: Full Sample 1957-2026", fontsize=16, pad=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Demeaned error-correction term", fontsize=12)
    ax.tick_params(axis="both", labelsize=10)
    ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.35)
    ax.spines["bottom"].set_alpha(0.35)
    ax.legend(
        handles=[
            Patch(facecolor=colors["Turbulent"], alpha=0.18, label="Turbulent month"),
            Patch(facecolor=colors["Normal"], alpha=0.18, label="Normal month"),
            *ax.get_legend_handles_labels()[0],
        ],
        loc="best",
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    data = add_demeaned_ect_and_regimes(add_raw_ect(load_data(INPUT_PATH)))
    train, test = split_data(data)

    plot_regime_chart(data, OUTPUT_PATH)
    print(f"Fixed training-set raw ECT mean used for demeaning: {TRAINING_ECT_MEAN:.6f}")
    print()
    print_ect_stats(train, test)
    print_monthly_classification(test)
    print_regime_summary(train, test)
    print(
        f"Training set: {train['date'].min():%Y-%m-%d} to {train['date'].max():%Y-%m-%d}, "
        f"{len(train)} rows"
    )
    print(
        f"Test set: {test['date'].min():%Y-%m-%d} to {test['date'].max():%Y-%m-%d}, "
        f"{len(test)} rows"
    )
    print(f"Chart saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
