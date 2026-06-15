from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


BASE_DIR = Path(__file__).resolve().parent
TRAIN_PATH = BASE_DIR / "fred_train.csv"
TEST_PATH = BASE_DIR / "fred_test.csv"
OUTPUT_PATH = BASE_DIR / "ect_full_sample.png"

REQUIRED_COLUMNS = ["date", "CPI", "CPI_core", "FPP"]
BETA_FPP = 0.736
GAMMA_THRESHOLD = -0.027


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


def add_ect(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data["ECT"] = np.log(data["CPI"]) - BETA_FPP * np.log(data["FPP"])
    return data


def print_ect_stats(label: str, data: pd.DataFrame) -> None:
    print(f"{label} ECT summary")
    print(f"Date range: {data['date'].min():%Y-%m-%d} to {data['date'].max():%Y-%m-%d}")
    print(f"Rows: {len(data)}")
    print(f"Min:  {data['ECT'].min():.6f}")
    print(f"Max:  {data['ECT'].max():.6f}")
    print(f"Mean: {data['ECT'].mean():.6f}")
    print()


def plot_ect(train: pd.DataFrame, test: pd.DataFrame, output_path: Path) -> None:
    if "seaborn-v0_8-whitegrid" in plt.style.available:
        plt.style.use("seaborn-v0_8-whitegrid")
    else:
        plt.style.use("default")

    full = pd.concat([train, test], ignore_index=True).sort_values("date")
    y_min = min(full["ECT"].min(), GAMMA_THRESHOLD, 0)
    y_max = max(full["ECT"].max(), GAMMA_THRESHOLD, 0)
    y_padding = (y_max - y_min) * 0.08 if y_max > y_min else 0.05

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axhspan(0, y_max + y_padding, color="#c7e9c0", alpha=0.22)
    ax.axhspan(y_min - y_padding, 0, color="#fcbba1", alpha=0.22)

    ax.plot(
        train["date"],
        train["ECT"],
        color="#808080",
        linewidth=1.7,
        label="Training ECT (Jan 1957-May 2022)",
    )
    ax.plot(
        test["date"],
        test["ECT"],
        color="#1f77b4",
        linewidth=2.2,
        label="Test ECT (Jun 2022-latest)",
    )
    ax.axhline(
        GAMMA_THRESHOLD,
        color="#8b0000",
        linestyle="--",
        linewidth=1.5,
        label=f"TVECM threshold gamma = {GAMMA_THRESHOLD:.3f}",
    )
    ax.axhline(
        0,
        color="#333333",
        linestyle="--",
        linewidth=1.2,
        label="Zero",
    )

    ax.set_title("Error Correction Term: Full Sample + Out-of-Sample", fontsize=15, pad=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("ECT = log(CPI) - 0.736 x log(FPP)", fontsize=12)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    ax.tick_params(axis="both", labelsize=10)
    ax.xaxis.set_major_locator(mdates.YearLocator(base=5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_alpha(0.35)
    ax.spines["bottom"].set_alpha(0.35)

    handles, labels = ax.get_legend_handles_labels()
    handles.extend(
        [
            Patch(facecolor="#c7e9c0", alpha=0.22, label="ECT positive: CPI above long-run relationship"),
            Patch(facecolor="#fcbba1", alpha=0.22, label="ECT negative: CPI below long-run relationship"),
        ]
    )
    ax.legend(handles=handles, loc="best", frameon=False, fontsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    train = add_ect(load_data(TRAIN_PATH))
    test = add_ect(load_data(TEST_PATH))

    plot_ect(train, test, OUTPUT_PATH)

    print_ect_stats("Training period", train)
    print_ect_stats("Test period", test)
    print(f"Chart saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
