# pip install fredapi pandas
#
# You need a free FRED API key. Request one at:
# https://fred.stlouisfed.org/docs/api/api_key.html
# Then provide it via the FRED_API_KEY environment variable, e.g.:
#   export FRED_API_KEY='your_api_key_here'

import os
import sys

import pandas as pd
from fredapi import Fred

API_KEY = os.environ.get("FRED_API_KEY")
if not API_KEY:
    sys.exit(
        "FRED_API_KEY is not set. Request a free key at "
        "https://fred.stlouisfed.org/docs/api/api_key.html and export it, e.g. "
        "`export FRED_API_KEY='...'`, before running this script."
    )

START_DATE = '1957-01-01'
END_DATE = '2022-05-31'

SERIES = {
    'CPIAUCSL': 'CPI',
    'CPILFESL': 'CPI_core',
    'PPIENG': 'FPP',
}


def main():
    fred = Fred(api_key=API_KEY)

    frames = []
    for series_id, column_name in SERIES.items():
        s = fred.get_series(
            series_id,
            observation_start=START_DATE,
            observation_end=END_DATE,
        )
        s = s.rename(column_name)
        frames.append(s)

    df = pd.concat(frames, axis=1)
    df.index = pd.to_datetime(df.index)
    df.index.name = 'date'
    df = df.loc[START_DATE:END_DATE]

    df.to_csv('fred_data.csv')
    print(f'Saved {len(df)} rows to fred_data.csv')


if __name__ == '__main__':
    main()
