"""
Era 1 (2009-2013) Sentiment Processing
--------------------------------------
Turns the raw PullPush search results collected by redditscraper_2009_2013.py
into the Era 1 sentiment dataset used by the analysis and figure scripts.

The PullPush collection searched all of Reddit for "<city>" AND "<keyword>",
so most raw results come from unrelated subreddits (e.g. r/hockey,
r/AskReddit, r/OnceUponATime for "Regina"). To make Era 1 comparable with
Eras 2-5, this script applies the same rules the 2014-2025 collectors use:

- Keep only posts from the seven study subreddits, mapped to cities
  (r/alberta -> Edmonton, r/saskatchewan -> Saskatoon, r/manitoba -> Winnipeg)
- Clean text and flag crime / weather keywords exactly as in
  redditscraper_2014_2019.py; keep posts matching either keyword set
- Score cleaned text with VADER (compound; negative <= -0.05,
  positive >= +0.05, neutral otherwise)

Outputs:
- data/sentiment/reddit_2009_2013_posts_sentiment.csv          (post level)
- data/sentiment/reddit_climate_crime_2009_2013_sentiment.csv  (city-month)
  (both mirrored to raw_data/Master_Data/)

Usage:
    python sentiment_2009_2013.py
"""
import re
from pathlib import Path

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_POSTS = (REPO_ROOT / 'raw_data' / 'social_media' / '2009-2013_sentiment'
             / 'era1_2009_2013_posts_with_sentiment.csv')
OUT_DIRS = [REPO_ROOT / 'data' / 'sentiment', REPO_ROOT / 'raw_data' / 'Master_Data']
POSTS_FILENAME = 'reddit_2009_2013_posts_sentiment.csv'
MONTHLY_FILENAME = 'reddit_climate_crime_2009_2013_sentiment.csv'

CITY_MAP = {
    'edmonton': 'Edmonton', 'alberta': 'Edmonton',
    'saskatoon': 'Saskatoon', 'saskatchewan': 'Saskatoon',
    'regina': 'Regina',
    'winnipeg': 'Winnipeg', 'manitoba': 'Winnipeg',
}

# Same keyword sets as redditscraper_2014_2019.py
CRIME_KEYWORDS = [
    'crime', 'theft', 'break-in', 'break in', 'burglary', 'assault',
    'robbery', 'vandalism', 'police', 'cops', 'rcmp', 'safety',
    'dangerous', 'stolen', 'stole', 'carjack', 'mugging', 'murder',
    'shooting', 'stabbing', 'domestic violence', 'gang', 'drug deal',
    'arrest', 'criminal', 'victim', 'attacked', 'security', 'violence',
    'homicide', 'fraud', 'scam', 'harassment', 'break and enter'
]

WEATHER_KEYWORDS = [
    'cold', 'hot', 'weather', 'temperature', 'snow', 'blizzard',
    'heat wave', 'heatwave', 'winter', 'summer', 'freeze', 'freezing',
    'chinook', 'windchill', 'frost', 'ice storm', 'precipitation',
    'celsius', 'fahrenheit', 'warm', 'humid', 'drought', 'rain',
    'storm', 'climate', 'seasonal', 'spring', 'fall', 'autumn',
    'extreme cold', 'extreme heat', 'tornado', 'flooding', 'wildfire'
]


def clean_text(text):
    """Clean text for sentiment analysis (same rules as reddit_common.py)."""
    if pd.isna(text):
        return ''
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
    text = re.sub(r'\d+\s+[\w\s]+\s+(street|st|avenue|ave|road|rd|drive|dr|boulevard|blvd)',
                  '[ADDRESS]', text, flags=re.IGNORECASE)
    text = re.sub(r'[^a-zA-Z\s\.\!\?]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def categorize(compound):
    if compound >= 0.05:
        return 'positive'
    if compound <= -0.05:
        return 'negative'
    return 'neutral'


def build_posts():
    """Filter raw PullPush results to study subreddits and score with VADER."""
    df = pd.read_csv(RAW_POSTS)
    df['city'] = df['subreddit'].str.lower().map(CITY_MAP)
    df = df[df['city'].notna()].drop_duplicates(subset=['id']).copy()

    df['full_text'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    df['cleaned_text'] = df['full_text'].apply(clean_text)
    df['contains_crime_keywords'] = df['full_text'].str.contains(
        '|'.join(CRIME_KEYWORDS), case=False, na=False, regex=True)
    df['contains_weather_keywords'] = df['full_text'].str.contains(
        '|'.join(WEATHER_KEYWORDS), case=False, na=False, regex=True)
    df = df[df['contains_crime_keywords'] | df['contains_weather_keywords']].copy()

    analyzer = SentimentIntensityAnalyzer()
    df['sentiment_compound'] = df['cleaned_text'].apply(
        lambda t: analyzer.polarity_scores(t)['compound'] if t else 0.0)
    df['sentiment_category'] = df['sentiment_compound'].apply(categorize)

    # redditscraper_2009_2013.py stores created_utc as a datetime string
    df['created_date'] = pd.to_datetime(df['created_utc'])
    df['year_month'] = df['created_date'].dt.strftime('%Y-%m')
    return df[['id', 'subreddit', 'city', 'created_date', 'year_month',
               'contains_crime_keywords', 'contains_weather_keywords',
               'sentiment_compound', 'sentiment_category']]


def summarise(group):
    """City-month summary with the same columns as the original Era 1 file."""
    def stats(g, suffix):
        n = len(g)
        cat = g['sentiment_category']
        return {
            f'n_posts{suffix}': n,
            f'mean_polarity{suffix}': g['sentiment_compound'].mean() if n else float('nan'),
            f'median_polarity{suffix}': g['sentiment_compound'].median() if n else float('nan'),
            f'pct_negative{suffix}': (cat == 'negative').mean() if n else float('nan'),
            f'pct_positive{suffix}': (cat == 'positive').mean() if n else float('nan'),
            f'pct_neutral{suffix}': (cat == 'neutral').mean() if n else float('nan'),
        }
    crime = group[group['contains_crime_keywords']]
    return pd.Series({**stats(group, ''), **stats(crime, '_crime')})


def main():
    posts = build_posts()
    monthly = (posts.groupby(['city', 'year_month'])
               .apply(summarise, include_groups=False)
               .reset_index()
               .astype({'n_posts': int, 'n_posts_crime': int}))

    for out_dir in OUT_DIRS:
        posts.to_csv(out_dir / POSTS_FILENAME, index=False)
        monthly.to_csv(out_dir / MONTHLY_FILENAME, index=False)

    print(f"Era 1 posts kept: {len(posts)}")
    print(posts.groupby('city').agg(
        n=('id', 'size'),
        months=('year_month', 'nunique'),
        mean_compound=('sentiment_compound', 'mean'),
        pct_crime=('contains_crime_keywords', 'mean'),
    ).round(3))
    print(f"Wrote {POSTS_FILENAME} and {MONTHLY_FILENAME} to: "
          + ', '.join(str(d) for d in OUT_DIRS))


if __name__ == '__main__':
    main()
