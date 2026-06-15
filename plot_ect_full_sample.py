#!/usr/bin/env python3

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BETA = 0.736
THRESHOLD_GAMMA = -0.027

BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / "fred_train.csv"
TEST_FILE = BASE_DIR / "fred_test.csv"
OUTPUT_FILE = BASE_DIR / "ect_full_sample.png"

REQUIRED_COLUMNS = {"date", "CPI", "FPP"}


def load_and_compute_ect(path: Path, label: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing_columns)}")

    df = df.sort_values("date").copy()

    if (df["CPI"] <= 0).any() or (df["FPP"] <= 0).any():
        raise ValueError(f"{path.name} contains non-positive CPI or FPP values, so logs cannot be computed.")

    df["ECT"] = np.log(df["CPI"]) - BETA * np.log(df["FPP"])
    df["period"] = label
    return df[["date", "CPI", "FPP", "ECT", "period"]]


def print_ect_stats(df: pd.DataFrame, label: str) -> None:
    stats = df["ECT"].agg(["min", "max", "mean"])
    start_date = df["date"].min().date()
    end_date = df["date"].max().date()

    print(f"\n{label} ECT statistics ({start_date} to {end_date})")
    print("-" * 56)
    print(f"Min : {stats['min']:.6f}")
    print(f"Max : {stats['max']:.6f}")
    print(f"Mean: {stats['mean']:.6f}")


def plot_ect(train: pd.DataFrame, test: pd.DataFrame) -> None:
    combined = pd.concat([train, test], ignore_index=True).sort_values("date")

    fig, ax = plt.subplots(figsize=(13, 7))

    ax.fill_between(
        combined["date"],
        combined["ECT"],
        0,
        where=combined["ECT"] >= 0,
        color="green",
        alpha=0.12,
        interpolate=True,
        label="ECT positive: CPI above long-run relationship",
    )
    ax.fill_between(
        combined["date"],
        combined["ECT"],
        0,
        where=combined["ECT"] < 0,
        color="red",
        alpha=0.12,
        interpolate=True,
        label="ECT negative: CPI below long-run relationship",
    )

    ax.plot(
        train["date"],
        train["ECT"],
        color="grey",
        linewidth=1.8,
        label="Training ECT (Jan 1957 to May 2022)",
    )
    ax.plot(
        test["date"],
        test["ECT"],
        color="royalblue",
        linewidth=2.2,
        label="Test ECT (Jun 2022 to latest)",
    )

    ax.axhline(
        THRESHOLD_GAMMA,
        color="firebrick",
        linestyle="--",
        linewidth=1.6,
        label=f"Threshold gamma = {THRESHOLD_GAMMA}",
    )
    ax.axhline(
        0,
        color="black",
        linestyle="--",
        linewidth=1.2,
        label="Zero",
    )

    ax.set_title("Error Correction Term: Full Sample + Out-of-Sample", fontsize=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("ECT = log(CPI) - 0.736 * log(FPP)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.autofmt_xdate()

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, dpi=300)
    plt.close(fig)


def main() -> None:
    train = load_and_compute_ect(TRAIN_FILE, "Training")
    test = load_and_compute_ect(TEST_FILE, "Test")

    print_ect_stats(train, "Training")
    print_ect_stats(test, "Test")

    plot_ect(train, test)
    print(f"\nSaved chart to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
