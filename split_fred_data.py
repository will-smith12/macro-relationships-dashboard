from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "fred_data_extended.csv"
TRAIN_PATH = BASE_DIR / "fred_train.csv"
TEST_PATH = BASE_DIR / "fred_test.csv"

REQUIRED_COLUMNS = ["date", "CPI", "CPI_core", "FPP"]
TRAIN_START = pd.Timestamp("1957-01-01")
TRAIN_END = pd.Timestamp("2022-05-01")
TEST_START = pd.Timestamp("2022-06-01")


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path, parse_dates=["date"])
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"{path.name} is missing required columns: {missing_columns}")

    return data[REQUIRED_COLUMNS].sort_values("date").reset_index(drop=True)


def save_split(data: pd.DataFrame, path: Path) -> None:
    data.to_csv(path, index=False, date_format="%Y-%m-%d")


def print_summary(name: str, path: Path, data: pd.DataFrame) -> None:
    if data.empty:
        raise ValueError(f"{name} split is empty")

    start_date = data["date"].min().strftime("%Y-%m-%d")
    end_date = data["date"].max().strftime("%Y-%m-%d")

    print(f"{name}: saved to {path}")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Rows: {len(data)}")
    print("First 3 rows:")
    print(data.head(3).to_string(index=False))
    print("Last 3 rows:")
    print(data.tail(3).to_string(index=False))
    print()


def main() -> None:
    data = load_data(INPUT_PATH)

    train = data[(data["date"] >= TRAIN_START) & (data["date"] <= TRAIN_END)].copy()
    test = data[data["date"] >= TEST_START].copy()

    save_split(train, TRAIN_PATH)
    save_split(test, TEST_PATH)

    print_summary("fred_train.csv", TRAIN_PATH, train)
    print_summary("fred_test.csv", TEST_PATH, test)


if __name__ == "__main__":
    main()
