"""
================================================================================
Urban Informatics — Statistical Analysis Script
================================================================================
Runs all statistical tests reported in the article:

    Test 1 — Kruskal-Wallis H test (inter-city polarity distributions)
    Test 2 — Mann-Whitney U with Bonferroni correction (pairwise city comparisons)
    Test 3 — Chi-square test of independence (categorical sentiment composition)
    Test 4 — Mann-Whitney U (crime vs. non-crime sentiment divergence)
    Test 5a — Pearson correlations (bivariate climate-crime associations)
    Test 5b — OLS regression with Durbin-Watson (multivariate climate-crime model)
    Test 5c — Trend robustness (detrended and first-difference correlations)

Tests 1–4 are run on Eras 2–5. Every era uses the same inclusion rule: posts
matching the crime or weather keyword sets. The 2020–2021 collector also kept
pandemic-keyword posts; those are excluded so Era 3 is comparable. Era 1
(2009–2013) is too sparse for inferential testing and is described only.

Requirements:
    pip install pandas numpy scipy

Data files required (read from the repository's data/ folder):
    reddit_climate_crime_2014_2019_sentiment.csv
    reddit_climate_crime_2020_2021_sentiment.csv
    reddit_climate_crime_2022_2023_sentiment.csv
    reddit_climate_crime_2024_2025_sentiment.csv
    Actual_Incidents_Edmonton.csv
    Actual_Incidents_Regina.csv
    Actual_Incidents_Winnipeg.csv
    Actual_Incidents_incidents Saskatoon.csv
    Master_All_Cities_Yearly_Climate_2010_to_2025_imputed.csv

Usage:
    python statistical_analysis.py

Output:
    Prints all results to console with clear section headers.
================================================================================
"""

import warnings
from itertools import combinations
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency, pearsonr, t as t_dist
warnings.filterwarnings('ignore')

# ── Directories ───────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[2] / 'data'
SENTIMENT_DIR = DATA_DIR / 'sentiment'
CRIME_DIR = DATA_DIR / 'crime'
CLIMATE_DIR = DATA_DIR / 'climate'

# ── Constants ─────────────────────────────────────────────────────────────────
CITIES = ['Edmonton', 'Saskatoon', 'Regina', 'Winnipeg']

CITY_MAP = {
    'edmonton':    'Edmonton',  'alberta':      'Edmonton',
    'saskatoon':   'Saskatoon', 'saskatchewan': 'Saskatoon',
    'regina':      'Regina',
    'winnipeg':    'Winnipeg',  'manitoba':     'Winnipeg',
}

VIOLENT_VIOLATION = 'Total violent Criminal Code violations [100]'

# Temperature_C_max is the annual mean of daily maximum temperatures
CLIM_VARS = [
    ('Relative_Humidity_Pct_mean', 'Mean Humidity (%)'),
    ('Cooling_Degree_Days_sum',    'Cooling Degree Days'),
    ('Temperature_C_max',          'Mean Daily Max Temp (°C)'),
]

CRIME_FILES = {
    'Edmonton':  CRIME_DIR / 'Actual_Incidents_Edmonton.csv',
    'Regina':    CRIME_DIR / 'Actual_Incidents_Regina.csv',
    'Winnipeg':  CRIME_DIR / 'Actual_Incidents_Winnipeg.csv',
    'Saskatoon': CRIME_DIR / 'Actual_Incidents_incidents Saskatoon.csv',
}

ERA_FILES = {
    'Era 2 (2014–2019)': SENTIMENT_DIR / 'reddit_climate_crime_2014_2019_sentiment.csv',
    'Era 3 (2020–2021)': SENTIMENT_DIR / 'reddit_climate_crime_2020_2021_sentiment.csv',
    'Era 4 (2022–2023)': SENTIMENT_DIR / 'reddit_climate_crime_2022_2023_sentiment.csv',
    'Era 5 (2024–2025)': SENTIMENT_DIR / 'reddit_climate_crime_2024_2025_sentiment.csv',
}


# ── Data loader ───────────────────────────────────────────────────────────────
def load_era(path):
    """
    Load a post-level sentiment CSV, map subreddits to canonical city names,
    and keep posts matching the crime or weather keyword sets (drops the
    pandemic-only posts the 2020–2021 collector added).
    """
    df = pd.read_csv(path)
    df['city'] = df['subreddit'].str.lower().map(CITY_MAP)
    in_scope = df['contains_crime_keywords'] | df['contains_weather_keywords']
    return df[df['city'].isin(CITIES) & in_scope].copy()


def load_eras(city_subreddits_only=False):
    """Load Eras 2–5; optionally drop the provincial subreddits (sensitivity check)."""
    eras = {era: load_era(path) for era, path in ERA_FILES.items()}
    if city_subreddits_only:
        eras = {era: df[df['subreddit'].str.lower() == df['city'].str.lower()]
                for era, df in eras.items()}
    return eras


def load_crime():
    """Load annual violent crime counts for all four cities."""
    out = {}
    for city, path in CRIME_FILES.items():
        df = pd.read_csv(path)
        sub = (df[df['Violations'] == VIOLENT_VIOLATION][['REF_DATE', 'VALUE']]
               .rename(columns={'REF_DATE': 'year', 'VALUE': 'violent_crime'}))
        sub['year'] = sub['year'].astype(int)
        out[city] = sub.set_index('year')['violent_crime']
    return out


def load_climate():
    """Load yearly imputed climate data for all four cities (2010–2024)."""
    df = pd.read_csv(
        CLIMATE_DIR / 'Master_All_Cities_Yearly_Climate_2010_to_2025_imputed.csv'
    )
    df['year'] = pd.to_datetime(df['Date']).dt.year
    return df[df['year'] <= 2024]


def city_crime_climate(crime, climate, city):
    """Join one city's annual violent crime counts to its yearly climate."""
    cl = climate[climate['City'] == city]
    return crime[city].reset_index().merge(cl, on='year').dropna()


# ── Significance formatter ────────────────────────────────────────────────────
def sig(p):
    if p < 0.001:
        return '***'
    if p < 0.01:
        return '**'
    if p < 0.05:
        return '*'
    if p < 0.10:
        return '†'
    return 'ns'


# ════════════════════════════════════════════════════════════════════════════
#  TEST 1 — KRUSKAL-WALLIS H TEST
#  Do sentiment polarity distributions differ across the four cities?
# ════════════════════════════════════════════════════════════════════════════
def test1_kruskal_wallis(eras):
    print("=" * 65)
    print("TEST 1 — KRUSKAL-WALLIS H TEST")
    print("Non-parametric test: do sentiment polarity distributions")
    print("differ significantly across the four cities?")
    print("-" * 65)
    print(f"{'Era':<22} {'H statistic':>13} {'p-value':>10} {'Sig':>5}")
    print("-" * 65)

    results = {}
    for era, df in eras.items():
        groups = [df[df['city'] == c]['sentiment_compound'].dropna().values
                  for c in CITIES]
        h, p = kruskal(*groups)
        results[era] = {'H': h, 'p': p, 'sig': sig(p)}
        print(f"{era:<22} {h:>13.3f} {p:>10.5f} {sig(p):>5}")

    print("-" * 65)
    print("Note: Era 1 (2009–2013) has 14–136 posts per city and is")
    print("      described only, not tested.")
    print()
    return results


# ════════════════════════════════════════════════════════════════════════════
#  TEST 2 — MANN-WHITNEY U WITH BONFERRONI CORRECTION
#  Which specific city pairs drive the Kruskal-Wallis result?
# ════════════════════════════════════════════════════════════════════════════
def test2_mannwhitney(eras):
    print("=" * 65)
    print("TEST 2 — MANN-WHITNEY U (pairwise + Bonferroni correction)")
    print("Which specific city pairs drive the Kruskal-Wallis result?")
    print()

    pairs = list(combinations(CITIES, 2))
    n_tests = len(pairs)  # 6 pairs → Bonferroni multiplier = 6
    results = {}

    for era, df in eras.items():
        print(f"  {era}")
        print(f"  {'City A':<12} {'City B':<12} {'U':>10} "
              f"{'p (raw)':>10} {'p (Bonf)':>10} {'Sig':>5}")
        print("  " + "-" * 60)
        era_results = []
        for c1, c2 in pairs:
            g1 = df[df['city'] == c1]['sentiment_compound'].dropna().values
            g2 = df[df['city'] == c2]['sentiment_compound'].dropna().values
            u, p = mannwhitneyu(g1, g2, alternative='two-sided')
            p_adj = min(p * n_tests, 1.0)
            flag = ' ◄' if p_adj < 0.05 else ''
            era_results.append({
                'c1': c1, 'c2': c2, 'U': u,
                'p_raw': p, 'p_bonf': p_adj, 'sig': sig(p_adj)
            })
            print(f"  {c1:<12} {c2:<12} {u:>10.0f} "
                  f"{p:>10.4f} {p_adj:>10.4f} {sig(p_adj):>5}{flag}")
        results[era] = era_results
        print()

    return results


# ════════════════════════════════════════════════════════════════════════════
#  TEST 3 — CHI-SQUARE TEST OF INDEPENDENCE
#  Do cities differ in their proportional mix of negative/neutral/positive?
# ════════════════════════════════════════════════════════════════════════════
def test3_chisquare(eras):
    print("=" * 65)
    print("TEST 3 — CHI-SQUARE TEST OF INDEPENDENCE")
    print("Do cities differ in the proportional mix of")
    print("negative / neutral / positive sentiment posts?")
    print("-" * 65)
    print(f"{'Era':<22} {'χ²':>9} {'df':>4} {'p-value':>10} {'Sig':>5}")
    print("-" * 65)

    results = {}
    for era, df in eras.items():
        table = np.array([
            [(df[df['city'] == c]['sentiment_category'] == cat).sum()
             for cat in ['negative', 'neutral', 'positive']]
            for c in CITIES
        ])
        chi2, p, dof, _ = chi2_contingency(table)
        results[era] = {'chi2': chi2, 'p': p, 'dof': dof, 'sig': sig(p)}
        print(f"{era:<22} {chi2:>9.3f} {dof:>4} {p:>10.5f} {sig(p):>5}")

    print("-" * 65)
    print()

    print("Contingency tables (raw counts) per era:")
    for era, df in eras.items():
        print(f"\n  {era}")
        print(f"  {'City':<12} {'Negative':>10} {'Neutral':>10} "
              f"{'Positive':>10} {'Total':>8}")
        print("  " + "-" * 52)
        for city in CITIES:
            c = df[df['city'] == city]
            neg = (c['sentiment_category'] == 'negative').sum()
            neu = (c['sentiment_category'] == 'neutral').sum()
            pos = (c['sentiment_category'] == 'positive').sum()
            print(f"  {city:<12} {neg:>10} {neu:>10} {pos:>10} {len(c):>8}")
    print()
    return results


# ════════════════════════════════════════════════════════════════════════════
#  TEST 4 — MANN-WHITNEY U: CRIME vs. NON-CRIME
#  Is crime-relevant sentiment significantly more negative than non-crime?
#  Crime and non-crime posts are independent samples, so a two-sample rank
#  test on all posts is used (no subsampling or pairing).
# ════════════════════════════════════════════════════════════════════════════
def test4_crime_divergence(eras):
    print("=" * 65)
    print("TEST 4 — MANN-WHITNEY U (crime vs. non-crime posts)")
    print("Is crime-relevant sentiment significantly more negative")
    print("than non-crime (weather) sentiment within each city?")

    results = {}

    for era, df in eras.items():
        print(f"\n  {era}")
        print(f"  {'City':<12} {'n crime':>8} {'n non':>7} {'U':>9} {'U/n₁n₂':>7} "
              f"{'p-value':>10} {'Sig':>5} {'Crime μ':>8} {'Non μ':>7} {'Δ':>7}")
        print("  " + "-" * 88)
        era_results = {}
        for city in CITIES:
            c = df[df['city'] == city]
            crime = c[c['contains_crime_keywords'].eq(True)]['sentiment_compound'].dropna().values
            noncrime = c[c['contains_crime_keywords'].eq(False)]['sentiment_compound'].dropna().values
            u, p = mannwhitneyu(crime, noncrime, alternative='two-sided')
            cles = u / (len(crime) * len(noncrime))
            diff = crime.mean() - noncrime.mean()
            era_results[city] = {
                'U': u, 'p': p, 'sig': sig(p), 'cles': cles,
                'n_crime': len(crime), 'n_non': len(noncrime),
                'crime_mean': crime.mean(),
                'non_mean':   noncrime.mean(),
                'delta':      diff,
            }
            print(f"  {city:<12} {len(crime):>8} {len(noncrime):>7} {u:>9.0f} {cles:>7.3f} "
                  f"{p:>10.5f} {sig(p):>5} {crime.mean():>8.3f} {noncrime.mean():>7.3f} {diff:>7.3f}")
        results[era] = era_results

    print()
    print("U/n₁n₂ = probability that a crime post scores higher than a")
    print("non-crime post (0.5 = no difference).")
    print()
    return results


# ════════════════════════════════════════════════════════════════════════════
#  TEST 5a — PEARSON CORRELATIONS
#  Bivariate climate-crime associations (n=15 per city, 2010–2024)
# ════════════════════════════════════════════════════════════════════════════
def test5a_pearson(crime, climate):
    print("=" * 65)
    print("TEST 5a — PEARSON CORRELATIONS")
    print("Bivariate climate vs. annual violent crime (n=15, 2010–2024)")
    print("-" * 65)
    print(f"{'City':<12} {'Variable':<26} {'r':>8} {'p-value':>10} {'Sig':>5}")
    print("-" * 65)

    results = {}
    for city in CITIES:
        merged = city_crime_climate(crime, climate, city)
        results[city] = {}
        for var, label in CLIM_VARS:
            r, p = pearsonr(merged[var], merged['violent_crime'])
            results[city][var] = {'r': r, 'p': p, 'sig': sig(p)}
            print(f"{city:<12} {label:<26} {r:>8.3f} {p:>10.4f} {sig(p):>5}")
        print()

    # Winnipeg humidity is entirely imputed for 2010–2012 (and 94% for 2013)
    wpg = city_crime_climate(crime, climate, 'Winnipeg')
    wpg = wpg[wpg['year'] >= 2014]
    r, p = pearsonr(wpg['Relative_Humidity_Pct_mean'], wpg['violent_crime'])
    print(f"Sensitivity — Winnipeg humidity, observed years only (2014–2024, "
          f"n={len(wpg)}): r = {r:.3f}, p = {p:.4f}")
    print()
    print("80% power at n=15 requires |r| ≥ 0.67 at α=0.05 (|r| ≥ 0.76 at α=0.01).")
    print()
    return results


# ════════════════════════════════════════════════════════════════════════════
#  TEST 5b — OLS REGRESSION + DURBIN-WATSON
#  Multivariate climate-crime model (z-scored predictors)
# ════════════════════════════════════════════════════════════════════════════
def fit_ols(merged):
    """Fit OLS of violent crime on z-scored climate predictors; return n and fit stats."""
    # Z-score predictors
    x = merged[['Temperature_C_max',
                'Relative_Humidity_Pct_mean',
                'Cooling_Degree_Days_sum']].copy()
    for col in x.columns:
        x[col] = (x[col] - x[col].mean()) / x[col].std()

    y = merged['violent_crime'].values
    x_mat = np.column_stack([np.ones(len(x)), x.values])

    # OLS via least squares
    betas = np.linalg.lstsq(x_mat, y, rcond=None)[0]
    residuals = y - x_mat @ betas
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    n, k = len(y), x_mat.shape[1] - 1
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - k - 1)

    # Durbin-Watson statistic
    dw = np.sum(np.diff(residuals) ** 2) / ss_res

    return n, {'r2': r2, 'r2_adj': r2_adj, 'dw': dw, 'betas': betas}


def test5b_ols(crime, climate):
    print("=" * 65)
    print("TEST 5b — OLS REGRESSION (z-scored predictors)")
    print("Outcome: annual violent crime")
    print("Predictors: Max Temperature + Mean Humidity + Cooling Degree Days")
    print("-" * 65)

    results = {}

    for city in CITIES:
        n, fit = fit_ols(city_crime_climate(crime, climate, city))
        dw, betas = fit['dw'], fit['betas']
        dw_flag = '← moderate autocorrelation' if 1.0 <= dw < 2.0 else '← autocorrelation concern'
        results[city] = fit

        print(f"\n  {city}  (n={n})")
        print(f"  R²            = {fit['r2']:.3f}")
        print(f"  Adjusted R²   = {fit['r2_adj']:.3f}")
        print(f"  Durbin-Watson = {dw:.3f}  {dw_flag}")
        print(f"  Intercept     = {betas[0]:>10.1f}")
        print(f"  β Temperature = {betas[1]:>10.1f}  incidents per 1 SD increase")
        print(f"  β Humidity    = {betas[2]:>10.1f}  incidents per 1 SD increase")
        print(f"  β CDD         = {betas[3]:>10.1f}  incidents per 1 SD increase")

    print()
    print("Note: All results treated as exploratory associations.")
    print("      Serial autocorrelation (DW < 2.0) means standard errors")
    print("      are underestimated. HAC-corrected SE recommended for")
    print("      formal inference.")
    print()
    return results


# ════════════════════════════════════════════════════════════════════════════
#  TEST 5c — TREND ROBUSTNESS
#  Do the climate-crime correlations survive removal of shared time trends?
# ════════════════════════════════════════════════════════════════════════════
def detrended_r(x, y, years):
    """
    Correlation of x and y after removing a linear year trend from each
    (the partial correlation controlling for year; df = n − 3).
    """
    rx = x - np.polyval(np.polyfit(years, x, 1), years)
    ry = y - np.polyval(np.polyfit(years, y, 1), years)
    r = np.corrcoef(rx, ry)[0, 1]
    dof = len(x) - 3
    t_stat = r * np.sqrt(dof / (1 - r ** 2))
    return r, 2 * t_dist.sf(abs(t_stat), dof)


def test5c_trend_robustness(crime, climate):
    print("=" * 65)
    print("TEST 5c — TREND ROBUSTNESS")
    print("Correlations after removing linear year trends (detrended) and")
    print("on year-to-year changes (first differences, n=14)")
    print("-" * 65)
    print(f"{'City':<12} {'Variable':<26} {'r(year)':>8} {'r detr':>7} {'p':>7} "
          f"{'r diff':>7} {'p':>7}")
    print("-" * 65)

    results = {}
    for city in CITIES:
        merged = city_crime_climate(crime, climate, city)
        years = merged['year'].values.astype(float)
        y = merged['violent_crime'].values.astype(float)
        results[city] = {'crime_year_r': pearsonr(years, y)[0]}
        for var, label in CLIM_VARS:
            x = merged[var].values
            r_year = pearsonr(years, x)[0]
            r_det, p_det = detrended_r(x, y, years)
            r_diff, p_diff = pearsonr(np.diff(x), np.diff(y))
            results[city][var] = {'r_year': r_year, 'r_detrended': r_det,
                                  'p_detrended': p_det, 'r_diff': r_diff,
                                  'p_diff': p_diff}
            print(f"{city:<12} {label:<26} {r_year:>8.3f} {r_det:>7.3f} {p_det:>7.3f} "
                  f"{r_diff:>7.3f} {p_diff:>7.3f}")
        print(f"{'':<12} {'(violent crime vs. year)':<26} "
              f"{results[city]['crime_year_r']:>8.3f}")
        print()

    return results


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    """Load all data and run every test in article order."""
    print("=" * 65)
    print("Urban Informatics — Statistical Analysis")
    print("=" * 65)
    print()

    # Load data
    eras = load_eras()
    crime = load_crime()
    climate = load_climate()

    # Run all tests
    test1_kruskal_wallis(eras)
    test2_mannwhitney(eras)
    test3_chisquare(eras)
    test4_crime_divergence(eras)
    test5a_pearson(crime, climate)
    test5b_ols(crime, climate)
    test5c_trend_robustness(crime, climate)

    # Sensitivity: r/alberta, r/saskatchewan and r/manitoba are pooled with
    # Edmonton, Saskatoon and Winnipeg; Regina has no provincial subreddit.
    print("=" * 65)
    print("SENSITIVITY — CITY SUBREDDITS ONLY (provincial subreddits dropped)")
    city_only = load_eras(city_subreddits_only=True)
    test1_kruskal_wallis(city_only)
    test2_mannwhitney(city_only)
    test3_chisquare(city_only)

    print("=" * 65)
    print("All tests complete.")
    print("=" * 65)


if __name__ == '__main__':
    main()
