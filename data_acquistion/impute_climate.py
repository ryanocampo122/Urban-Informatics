"""
Climate Data Imputation (All Cities)
------------------------------------
Fills gaps in the cleaned daily climate master file and rebuilds the
monthly and yearly summaries from the imputed daily data.

Steps:
1. Drop variables with more than 70% missing daily values (Humidex)
2. Fill each remaining gap with the city's mean for that calendar month,
   computed across all years
3. Linearly interpolate (within city) anything still missing — only
   seasonal variables with whole calendar months absent, i.e. snow on
   ground and wind chill in summer
4. Aggregate the imputed daily data to monthly and yearly resolution using
   the aggregation rules in masterclimate_acquistion.py

Monthly and yearly values are computed from the imputed daily data so that
years with seasonal gaps (e.g. Edmonton humidity, Jan–Mar 2015) are not
biased toward the months that happen to have observations.

Input:
    raw_data/climate/master_cleaned/Master_All_Cities_Daily_Climate_2010_to_2025_cleaned.csv

Outputs (written to data/climate/, raw_data/Master_Data/ and
raw_data/climate/master_imputed/):
    Master_All_Cities_{Daily,Monthly,Yearly}_Climate_2010_to_2025_imputed.csv

Usage:
    python impute_climate.py
"""
from pathlib import Path

import pandas as pd

from masterclimate_acquistion import MONTHLY_YEARLY_AGG_RULES

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEANED_DAILY = (REPO_ROOT / 'raw_data' / 'climate' / 'master_cleaned'
                 / 'Master_All_Cities_Daily_Climate_2010_to_2025_cleaned.csv')
OUT_DIRS = [
    REPO_ROOT / 'data' / 'climate',
    REPO_ROOT / 'raw_data' / 'Master_Data',
    REPO_ROOT / 'raw_data' / 'climate' / 'master_imputed',
]
OUT_TEMPLATE = 'Master_All_Cities_{}_Climate_2010_to_2025_imputed.csv'
MAX_MISSING = 0.70


def impute_daily(df):
    """Drop sparse variables, then fill gaps with city-month means."""
    value_cols = [c for c in df.columns if c not in ('Date', 'City')]
    missing = df[value_cols].isna().mean()
    dropped = missing[missing > MAX_MISSING].index.tolist()
    kept = [c for c in value_cols if c not in dropped]
    print(f"Dropped (> {MAX_MISSING:.0%} missing): {dropped}")

    out = df[['Date', 'City'] + kept].copy()
    month = out['Date'].dt.month
    for col in kept:
        city_month_mean = out.groupby(['City', month])[col].transform('mean')
        out[col] = out[col].fillna(city_month_mean)
        out[col] = out.groupby('City')[col].transform(lambda s: s.interpolate())
    return out


def aggregate(daily, freq):
    """Aggregate imputed daily data per city at monthly ('ME') or yearly ('YE') resolution."""
    rules = {c: r for c, r in MONTHLY_YEARLY_AGG_RULES.items() if c in daily.columns}
    frames = []
    for city, g in daily.groupby('City', sort=False):
        agg = g.set_index('Date').resample(freq).agg(rules)
        agg.insert(0, 'City', city)
        frames.append(agg)
    return pd.concat(frames).reset_index()


def main():
    daily = pd.read_csv(CLEANED_DAILY, parse_dates=['Date'])
    imputed = impute_daily(daily)
    outputs = {
        'Daily': imputed,
        'Monthly': aggregate(imputed, 'ME'),
        'Yearly': aggregate(imputed, 'YE'),
    }
    for out_dir in OUT_DIRS:
        for name, frame in outputs.items():
            frame.to_csv(out_dir / OUT_TEMPLATE.format(name), index=False,
                         date_format='%Y-%m-%d')
    print("Wrote imputed daily, monthly and yearly files to: "
          + ', '.join(str(d) for d in OUT_DIRS))


if __name__ == '__main__':
    main()
