"""
================================================================================
Urban Informatics — Statistical Tests and Table Generation
================================================================================
Runs all six statistical tests reported in the article and prints
publication-ready formatted tables to the console.

Tables generated (numbered as in the article):
    Table 1 — Statistical testing framework rationale
    Table 2 — Era summary (descriptive inputs to the era framework)
    Table 3 — Kruskal-Wallis H test results by era
    Table 4 — Mann-Whitney U pairwise results (Bonferroni-corrected)
    Table 5 — Chi-square test + categorical composition by city and era
    Table 6 — Mann-Whitney U: crime vs. non-crime divergence
    Table 7 — Pearson correlations + trend robustness: climate vs. violent crime
    Table 8 — OLS regression: multivariate climate-crime model

Tests run:
    Test 1  — Kruskal-Wallis H (inter-city polarity distributions)
    Test 2  — Mann-Whitney U with Bonferroni correction (pairwise)
    Test 3  — Chi-square test of independence (categorical composition)
    Test 4  — Mann-Whitney U (crime vs. non-crime divergence)
    Test 5a — Pearson correlations (bivariate climate-crime)
    Test 5b — OLS regression + Durbin-Watson (multivariate climate-crime)
    Test 5c — Trend robustness (detrended and first-difference correlations)

Data loading, the era inclusion rule (crime- or weather-keyword posts) and
the trend-robustness helper are shared with statistical_analysis.py.

Requirements:
    pip install pandas numpy scipy

Data files required (read from the repository's data/ folder):
    reddit_2009_2013_posts_sentiment.csv          (Era 1 — post level, descriptive only)
    reddit_climate_crime_2014_2019_sentiment.csv  (Era 2 — post level)
    reddit_climate_crime_2020_2021_sentiment.csv  (Era 3 — post level)
    reddit_climate_crime_2022_2023_sentiment.csv  (Era 4 — post level)
    reddit_climate_crime_2024_2025_sentiment.csv  (Era 5 — post level)
    Actual_Incidents_Edmonton.csv
    Actual_Incidents_Regina.csv
    Actual_Incidents_Winnipeg.csv
    Actual_Incidents_incidents Saskatoon.csv
    Master_All_Cities_Yearly_Climate_2010_to_2025_imputed.csv

Usage:
    python run_statistical_tests.py
================================================================================
"""

import warnings
from itertools import combinations
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency, pearsonr

from statistical_analysis import CLIM_VARS, detrended_r, load_era
warnings.filterwarnings('ignore')

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parents[2] / 'data'
SENTIMENT_DIR = DATA_DIR / 'sentiment'
CRIME_DIR = DATA_DIR / 'crime'
CLIMATE_DIR = DATA_DIR / 'climate'

CITIES = ['Edmonton', 'Saskatoon', 'Regina', 'Winnipeg']

VIOLENT_VIOLATION = 'Total violent Criminal Code violations [100]'

ERA1_POSTS = SENTIMENT_DIR / 'reddit_2009_2013_posts_sentiment.csv'

ERA_FILES = {
    'Era 2 (2014–2019)':
        SENTIMENT_DIR / 'reddit_climate_crime_2014_2019_sentiment.csv',
    'Era 3 (2020–2021)':
        SENTIMENT_DIR / 'reddit_climate_crime_2020_2021_sentiment.csv',
    'Era 4 (2022–2023)':
        SENTIMENT_DIR / 'reddit_climate_crime_2022_2023_sentiment.csv',
    'Era 5 (2024–2025)':
        SENTIMENT_DIR / 'reddit_climate_crime_2024_2025_sentiment.csv',
}

CRIME_FILES = {
    'Edmonton':  CRIME_DIR / 'Actual_Incidents_Edmonton.csv',
    'Regina':    CRIME_DIR / 'Actual_Incidents_Regina.csv',
    'Winnipeg':  CRIME_DIR / 'Actual_Incidents_Winnipeg.csv',
    'Saskatoon': CRIME_DIR / 'Actual_Incidents_incidents Saskatoon.csv',
}


# ── Helpers ──────────────────────────────────────────────────────────────────
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


def divider(width=75):
    print('─' * width)


def header(title):
    print()
    print('═' * 75)
    print(title)
    print('═' * 75)


def load_crime_annual():
    out = {}
    for city, path in CRIME_FILES.items():
        df = pd.read_csv(path)
        sub = df[df['Violations'] == VIOLENT_VIOLATION][
            ['REF_DATE', 'VALUE']
        ].rename(columns={'REF_DATE': 'year', 'VALUE': 'violent_crime'})
        sub['year'] = sub['year'].astype(int)
        out[city] = sub.set_index('year')['violent_crime']
    return out


def load_climate():
    df = pd.read_csv(CLIMATE_DIR / 'Master_All_Cities_Yearly_Climate_'
                     '2010_to_2025_imputed.csv')
    df['year'] = pd.to_datetime(df['Date']).dt.year
    return df[df['year'] <= 2024]


def city_crime_climate(crime, climate, city):
    cl = climate[climate['City'] == city]
    return crime[city].reset_index().merge(cl, on='year').dropna()


def split_crime_sentiment(c):
    kw = c['contains_crime_keywords']
    return (c[kw.eq(True)]['sentiment_compound'].dropna().values,
            c[kw.eq(False)]['sentiment_compound'].dropna().values)


def fit_ols(merged):
    x = merged[['Temperature_C_max',
                'Relative_Humidity_Pct_mean',
                'Cooling_Degree_Days_sum']].copy()
    for col in x.columns:
        x[col] = (x[col] - x[col].mean()) / x[col].std()

    y = merged['violent_crime'].values
    x_mat = np.column_stack([np.ones(len(x)), x.values])
    betas = np.linalg.lstsq(x_mat, y, rcond=None)[0]
    residuals = y - x_mat @ betas
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res/ss_tot
    n, k = len(y), x_mat.shape[1]-1
    r2_adj = 1 - (1-r2)*(n-1)/(n-k-1)
    dw = np.sum(np.diff(residuals)**2) / ss_res
    return n, {'r2': r2, 'r2_adj': r2_adj, 'dw': dw, 'betas': betas}


# ── TABLE 1 — Statistical testing framework rationale ────────────────────────
def print_table4():
    header("TABLE 1 — Statistical Testing Framework Rationale")
    print(f"{'#':<4} {'Test':<30} {'Research Question':<35} {'Justification'}")
    divider()
    rows = [
        ('1',  'Kruskal-Wallis H',
         'Do polarity distributions differ across cities?',
         'Non-parametric; sentiment is non-normally distributed. '
         'Tests full distribution shape, not just means.'),
        ('2',  'Mann-Whitney U + Bonferroni',
         'Which city pairs drive the KW result?',
         'Pairwise follow-up. Bonferroni correction (×6) controls '
         'family-wise error rate.'),
        ('3',  'Chi-square independence',
         'Do cities differ in neg/neutral/pos composition?',
         'Categorical complement to Test 1. Cities may share means but '
         'differ in proportional mix.'),
        ('4',  'Mann-Whitney U (crime vs. non-crime)',
         'Is crime sentiment more negative than non-crime?',
         'Two-sample within-city test on all posts. Validates crime '
         'discourse as distinct from weather discourse.'),
        ('5a', 'Pearson correlation',
         'Are climate variables associated with crime?',
         'Pre-specified inclusion rule: |r|≥0.40 in ≥1 city AND '
         'theoretically grounded mechanism.'),
        ('5b', 'OLS regression (z-scored)',
         'Combined explanatory power of climate predictors?',
         'Z-scoring enables direct β comparison across variables in '
         'different units. DW flags autocorrelation.'),
        ('5c', 'Trend robustness',
         'Do associations survive removal of shared time trends?',
         'Detrended (partial on year) and first-difference '
         'correlations guard against spurious trend correlation.'),
    ]
    for num, test, question, justification in rows:
        print(f"\n  {num:<4} {test}")
        print(f"       Q: {question}")
        print(f"       J: {justification}")
    print()


# ── TEST 1 + TABLE 3 — Kruskal-Wallis ────────────────────────────────────────
def run_test1(eras):
    header("TEST 1 + TABLE 3 — Kruskal-Wallis H Test")
    print("Research question: Do sentiment polarity distributions differ")
    print("significantly across the four cities within each era?")
    print()
    print(f"{'Era':<22} {'H':>10} {'p-value':>10} {'Sig':>5}  Interpretation")
    divider()

    results = {}
    interps = {
        'Era 2 (2014–2019)': ('Significant — Regina outlier emerges '
                              '(mean +0.122 vs −0.045 to −0.089)'),
        'Era 3 (2020–2021)': ('NOT significant — pandemic homogenises '
                              'city profiles'),
        'Era 4 (2022–2023)': ('Strongest result in study — post-pandemic '
                              'divergence intensifies'),
        'Era 5 (2024–2025)': ('Significant but attenuated — high-positivity '
                              'convergence begins'),
    }
    for era, df in eras.items():
        groups = [df[df['city'] == c]['sentiment_compound'].dropna().values
                  for c in CITIES]
        h, p = kruskal(*groups)
        results[era] = {'H': h, 'p': p}
        print(f"  {era:<20} {h:>10.3f} {p:>10.5f} {sig(p):>5}  "
              f"{interps.get(era, '')}")
    divider()
    print("Note: Era 1 (2009–2013) is described only — 14–136 posts "
          "per city.")
    return results


# ── TEST 2 + TABLE 4 — Mann-Whitney U ────────────────────────────────────────
def run_test2(eras):
    header("TEST 2 + TABLE 4 — Mann-Whitney U (Bonferroni-corrected)")
    print("Research question: Which specific city pairs drive the "
          "Kruskal-Wallis result?")
    print("Bonferroni multiplier = 6 (all pairwise combinations of 4 cities)")
    print()

    pairs = list(combinations(CITIES, 2))
    results = {}

    for era, df in eras.items():
        print(f"  {era}")
        print(f"  {'City A':<12} {'City B':<12} {'U':>10} "
              f"{'p raw':>10} {'p Bonf':>10} {'Sig':>5}")
        divider(72)
        era_res = []
        for c1, c2 in pairs:
            g1 = df[df['city'] == c1]['sentiment_compound'].dropna().values
            g2 = df[df['city'] == c2]['sentiment_compound'].dropna().values
            u, p = mannwhitneyu(g1, g2, alternative='two-sided')
            p_adj = min(p * 6, 1.0)
            flag = ' ◄ SIGNIFICANT' if p_adj < 0.05 else ''
            era_res.append({'c1': c1, 'c2': c2, 'U': u, 'p_raw': p,
                            'p_bonf': p_adj, 'sig': sig(p_adj)})
            print(f"  {c1:<12} {c2:<12} {u:>10.0f} {p:>10.4f} "
                  f"{p_adj:>10.4f} {sig(p_adj):>5}{flag}")
        results[era] = era_res
        print()
    return results


# ── TEST 3 + TABLE 5 — Chi-square ────────────────────────────────────────────
def run_test3(eras):
    header("TEST 3 + TABLE 5 — Chi-Square Test of Independence")
    print("Research question: Do cities differ in categorical sentiment "
          "composition")
    print("(% negative / neutral / positive posts) beyond polarity intensity?")
    print()
    print(f"{'Era':<22} {'χ²':>9} {'df':>4} {'p-value':>10} {'Sig':>5}")
    divider()

    results = {}
    for era, df in eras.items():
        table = np.array([
            [(df[df['city'] == c]['sentiment_category'] == cat).sum()
             for cat in ['negative', 'neutral', 'positive']]
            for c in CITIES
        ])
        chi2, p, dof, _ = chi2_contingency(table)
        results[era] = {'chi2': chi2, 'p': p, 'dof': dof}
        print(f"  {era:<20} {chi2:>9.3f} {dof:>4} {p:>10.5f} {sig(p):>5}")

    divider()
    print("\nContingency tables — counts and percentages:")
    for era, df in eras.items():
        print(f"\n  {era}")
        print(f"  {'City':<12} {'N':>6} {'Negative':>10} "
              f"{'Neutral':>10} {'Positive':>10}")
        divider(54)
        for city in CITIES:
            c = df[df['city'] == city]
            neg = (c['sentiment_category'] == 'negative').sum()
            neu = (c['sentiment_category'] == 'neutral').sum()
            pos = (c['sentiment_category'] == 'positive').sum()
            n = len(c)
            print(f"  {city:<12} {n:>6} "
                  f"{neg:>5} ({neg/n*100:4.1f}%) "
                  f"{neu:>5} ({neu/n*100:4.1f}%) "
                  f"{pos:>5} ({pos/n*100:4.1f}%)")
    return results


# ── TEST 4 + TABLE 6 — Mann-Whitney U (crime vs. non-crime) ─────────────────
def run_test4(eras):
    header("TEST 4 + TABLE 6 — Mann-Whitney U: Crime vs. Non-Crime Posts")
    print("Research question: Is crime-relevant sentiment significantly more")
    print("negative than non-crime (weather) sentiment within each city?")
    print("(Crime and non-crime posts are independent samples; all posts used)")
    print()

    results = {}

    for era, df in eras.items():
        print(f"  {era}")
        print(f"  {'City':<12} {'n crime':>8} {'n non':>6} {'U':>8} {'U/n₁n₂':>7} "
              f"{'p-value':>10} {'Sig':>5} {'Crime μ':>8} {'Non μ':>7} {'Δ':>7}")
        divider(86)
        era_res = {}
        for city in CITIES:
            crime, noncrime = split_crime_sentiment(df[df['city'] == city])
            u, p = mannwhitneyu(crime, noncrime, alternative='two-sided')
            cles = u / (len(crime) * len(noncrime))
            diff = crime.mean() - noncrime.mean()
            era_res[city] = {'U': u, 'p': p, 'cles': cles,
                             'crime_mean': crime.mean(),
                             'non_mean': noncrime.mean(), 'delta': diff}
            flag = ' ◄' if p < 0.05 else ''
            print(f"  {city:<12} {len(crime):>8} {len(noncrime):>6} {u:>8.0f} "
                  f"{cles:>7.3f} {p:>10.5f} {sig(p):>5} {crime.mean():>8.3f} "
                  f"{noncrime.mean():>7.3f} {diff:>7.3f}{flag}")
        results[era] = era_res
        print()
    print("U/n₁n₂ = probability that a crime post scores higher than a "
          "non-crime post (0.5 = no difference).")
    return results


# ── TEST 5a + TABLE 7 — Pearson ──────────────────────────────────────────────
def run_test5a(crime, climate):
    header("TEST 5a + TABLE 7 — Pearson Correlations "
           "(Climate vs. Violent Crime)")
    print("Research question: Are annual climate variables associated with")
    print("annual violent crime counts at the city level? (n=15, 2010–2024)")
    print("Inclusion rule: peak |r| ≥ 0.40 in ≥1 city AND "
          "theoretically grounded.")
    print()
    print(f"{'City':<12} {'Variable':<26} {'r':>8} {'p-value':>10} {'Sig':>5}")
    divider()

    results = {}
    for city in CITIES:
        merged = city_crime_climate(crime, climate, city)
        results[city] = {}
        for var, label in CLIM_VARS:
            r, p = pearsonr(merged[var], merged['violent_crime'])
            results[city][var] = {'r': r, 'p': p, 'sig': sig(p)}
            flag = ' ◄' if p < 0.10 else ''
            print(f"  {city:<12} {label:<26} {r:>8.3f} {p:>10.4f} "
                  f"{sig(p):>5}{flag}")
        print()
    divider()
    wpg = city_crime_climate(crime, climate, 'Winnipeg')
    wpg = wpg[wpg['year'] >= 2014]
    r, p = pearsonr(wpg['Relative_Humidity_Pct_mean'], wpg['violent_crime'])
    print(f"Winnipeg humidity, observed years only (2014–2024, n={len(wpg)}; "
          f"2010–2013 imputed): r = {r:.3f}, p = {p:.4f}")
    print("80% power at n=15 requires |r| ≥ 0.67 at α=0.05 "
          "(|r| ≥ 0.76 at α=0.01).")
    print("See Test 5c: none of these associations survives detrending.")
    return results


# ── TEST 5b + TABLE 8 — OLS regression ──────────────────────────────────────
def run_test5b(crime, climate):
    header("TEST 5b + TABLE 8 — OLS Regression (z-scored predictors)")
    print("Research question: Combined explanatory power of temperature,")
    print("humidity, and CDD for annual violent crime. "
          "All predictors z-scored.")
    print()

    results = {}
    for city in CITIES:
        n, fit = fit_ols(city_crime_climate(crime, climate, city))
        dw, betas = fit['dw'], fit['betas']
        dw_note = ('moderate autocorrelation' if 1.0 <= dw < 2.0
                   else 'AUTOCORRELATION CONCERN')

        results[city] = fit

        print(f"  {city} (n={n})")
        print(f"  {'R²':<18} = {fit['r2']:.3f}")
        print(f"  {'Adjusted R²':<18} = {fit['r2_adj']:.3f}")
        print(f"  {'Durbin-Watson':<18} = {dw:.3f}  [{dw_note}]")
        print(f"  {'Intercept':<18} = {betas[0]:>10.1f}")
        print(f"  {'β Temperature':<18} = {betas[1]:>10.1f}  "
              "(incidents per 1 SD)")
        print(f"  {'β Humidity':<18} = {betas[2]:>10.1f}  "
              "(incidents per 1 SD)")
        print(f"  {'β CDD':<18} = {betas[3]:>10.1f}  "
              "(incidents per 1 SD)")
        print()

    divider()
    print("All results exploratory. DW < 1.0 indicates positive serial")
    print("autocorrelation — standard errors are underestimated.")
    print("HAC-corrected SE recommended for formal inference.")
    return results


# ── TEST 5c + TABLE 7 — Trend robustness ────────────────────────────────────
def run_test5c(crime, climate):
    header("TEST 5c + TABLE 7 — Trend Robustness (Climate vs. Violent Crime)")
    print("Research question: Do climate-crime correlations survive removal of")
    print("the shared linear time trend (detrended, df=12) and on year-to-year")
    print("changes (first differences, n=14)?")
    print()
    print(f"{'City':<12} {'Variable':<26} {'r(year)':>8} {'r detr':>7} "
          f"{'p':>7} {'r diff':>7} {'p':>7}")
    divider()

    results = {}
    for city in CITIES:
        merged = city_crime_climate(crime, climate, city)
        years = merged['year'].values.astype(float)
        y = merged['violent_crime'].values.astype(float)
        results[city] = {}
        for var, label in CLIM_VARS:
            x = merged[var].values
            r_det, p_det = detrended_r(x, y, years)
            r_diff, p_diff = pearsonr(np.diff(x), np.diff(y))
            results[city][var] = {'r_detrended': r_det, 'p_detrended': p_det,
                                  'r_diff': r_diff, 'p_diff': p_diff}
            print(f"  {city:<12} {label:<26} {pearsonr(years, x)[0]:>8.3f} "
                  f"{r_det:>7.3f} {p_det:>7.3f} {r_diff:>7.3f} {p_diff:>7.3f}")
        print(f"  {'':<12} {'(violent crime vs. year)':<26} "
              f"{pearsonr(years, y)[0]:>8.3f}")
        print()
    return results


# ── TABLE 2 inputs — Era summary ─────────────────────────────────────────────
def era_summary(df):
    """Per-city descriptive statistics for one era of post-level data."""
    df = df.copy()
    df['ym'] = pd.to_datetime(df['created_date'], utc=True).dt.to_period('M')
    rows = []
    for city in CITIES:
        c = df[df['city'] == city]
        monthly = c.groupby('ym')['sentiment_compound'].agg(['mean', 'size'])
        crime = c[c['contains_crime_keywords'].eq(True)]
        rows.append({
            'city': city, 'n': len(c), 'months': len(monthly),
            'posts_per_month': monthly['size'].median(),
            'mean': c['sentiment_compound'].mean(),
            'crime_mean': crime['sentiment_compound'].mean(),
            'pct_crime': c['contains_crime_keywords'].mean() * 100,
            'pct_neg': (c['sentiment_category'] == 'negative').mean() * 100,
            'crime_pct_neg': (crime['sentiment_category'] == 'negative').mean() * 100,
            'sd_monthly': monthly['mean'].std(),
        })
    return pd.DataFrame(rows).set_index('city')


def print_era_summary(eras):
    header("TABLE 2 INPUTS — Era Summary by City")
    print("n = posts; posts/mo = median posts per observed month;")
    print("SD monthly = standard deviation of monthly mean sentiment (Figure 3).")
    era1 = pd.read_csv(ERA1_POSTS)
    for era, df in {'Era 1 (2009–2013)': era1, **eras}.items():
        print(f"\n  {era}")
        print(era_summary(df).round(3).to_string())


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    print()
    print('█' * 75)
    print('  Urban Informatics — Statistical Tests and Table Generation')
    print('█' * 75)

    # Load data
    print("\nLoading data...")
    eras = {era: load_era(path) for era, path in ERA_FILES.items()}
    crime = load_crime_annual()
    climate = load_climate()
    print(f"  Eras loaded: "
          f"{', '.join(f'{e} (n={len(df)})' for e, df in eras.items())}")
    print(f"  Crime data: {list(crime.keys())}")
    print(f"  Climate data: {climate['City'].unique().tolist()}, "
          f"years {climate['year'].min()}–{climate['year'].max()}")

    # Run all tests
    print_table4()
    print_era_summary(eras)
    run_test1(eras)
    run_test2(eras)
    run_test3(eras)
    run_test4(eras)
    run_test5a(crime, climate)
    run_test5b(crime, climate)
    run_test5c(crime, climate)

    print()
    print('█' * 75)
    print('  All tests complete.')
    print('█' * 75)
    print()


if __name__ == '__main__':
    main()
