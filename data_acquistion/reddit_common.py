"""
Shared processing steps for the PRAW-based Reddit collectors
------------------------------------------------------------
Used by redditscraper_2014_2019.py, redditscraper_2020_2021.py and
redditscraper_2022_2023.py, whose collector classes inherit from
RedditCollectorBase and set:

    START_YEAR, END_YEAR   default window for generate_month_list()
    GIS_FILENAME           default output file for export_for_gis()
    GIS_FLAG_COLUMNS       era-specific keyword/flag columns kept in the GIS export
"""
import hashlib
import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)

# City coordinates for mapping (approximate city centers)
CITY_COORDS = {
    'Saskatoon': {'lat': 52.1332, 'lon': -106.6700, 'province': 'Saskatchewan'},
    'Regina': {'lat': 50.4452, 'lon': -104.6189, 'province': 'Saskatchewan'},
    'Winnipeg': {'lat': 49.8951, 'lon': -97.1384, 'province': 'Manitoba'},
    'Edmonton': {'lat': 53.5461, 'lon': -113.4938, 'province': 'Alberta'},
    'saskatchewan': {'lat': 52.9399, 'lon': -106.4509, 'province': 'Saskatchewan'},
    'manitoba': {'lat': 49.8951, 'lon': -97.1384, 'province': 'Manitoba'},
    'alberta': {'lat': 53.9333, 'lon': -116.5765, 'province': 'Alberta'}
}


class RedditCollectorBase:
    """Era-independent steps shared by the 2014–2023 Reddit collectors."""

    START_YEAR = None
    END_YEAR = None
    GIS_FILENAME = None
    GIS_FLAG_COLUMNS = ()
    sentiment_analyzer = None  # set by each collector's __init__

    def generate_month_list(self, start_year=None, start_month=1,
                            end_year=None, end_month=12):
        """
        Generate list of (year, month) tuples for the collector's era
        (START_YEAR-END_YEAR unless other bounds are given)

        Returns:
            List of (year, month) tuples
        """
        start_year = self.START_YEAR if start_year is None else start_year
        end_year = self.END_YEAR if end_year is None else end_year
        months = []
        current_year = start_year
        current_month = start_month

        while (current_year < end_year) or (current_year == end_year and current_month <= end_month):
            months.append((current_year, current_month))
            current_month += 1
            if current_month > 12:
                current_month = 1
                current_year += 1

        logger.info(f"Generated {len(months)} months from {start_year}-{start_month:02d} "
                    f"to {end_year}-{end_month:02d}")
        return months

    def _extract_post_data(self, submission):
        """Extract relevant data from a Reddit submission"""
        return {
            'id': submission.id,
            'title': submission.title,
            'selftext': submission.selftext,
            'created_utc': submission.created_utc,
            'author': str(submission.author) if submission.author else '[deleted]',
            'score': submission.score,
            'num_comments': submission.num_comments,
            'link_flair_text': submission.link_flair_text,
            'upvote_ratio': submission.upvote_ratio,
            'permalink': submission.permalink,
            'url': submission.url,
            'is_self': submission.is_self,
            'over_18': submission.over_18,
            'spoiler': submission.spoiler,
            'stickied': submission.stickied,
            'locked': submission.locked
        }

    def _clean_data(self, df):
        """Clean and standardize data"""
        # Remove deleted/removed content
        df = df[~df['selftext'].isin(['[deleted]', '[removed]'])]
        df = df[~df['author'].isin(['[deleted]', 'AutoModerator', '[removed]'])]

        # Handle missing values
        df['selftext'] = df['selftext'].fillna('')
        df['title'] = df['title'].fillna('')
        df['link_flair_text'] = df['link_flair_text'].fillna('None')

        # Create full text field
        df['full_text'] = df['title'] + ' ' + df['selftext']

        # Remove duplicates
        df = df.drop_duplicates(subset=['id'])

        # Convert timestamps
        df['created_utc'] = pd.to_numeric(df['created_utc'], errors='coerce')
        df['created_date'] = pd.to_datetime(df['created_utc'], unit='s', utc=True)

        # Remove posts with invalid dates
        df = df.dropna(subset=['created_date'])

        return df

    def _clean_text(self, text):
        """Clean text for sentiment analysis"""
        if pd.isna(text):
            return ''

        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        text = re.sub(r'\d+\s+[\w\s]+\s+(street|st|avenue|ave|road|rd|drive|dr|boulevard|blvd)',
                      '[ADDRESS]', text, flags=re.IGNORECASE)
        text = re.sub(r'[^a-zA-Z\s\.\!\?]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def _anonymize_data(self, df):
        """Anonymize author information"""
        df['author_hash'] = df['author'].apply(
            lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:16]
        )
        df = df.drop('author', axis=1)
        return df

    def _add_sentiment(self, df):
        """Add sentiment scores using VADER"""
        def get_sentiment_scores(text):
            if pd.isna(text) or text == '':
                return {
                    'sentiment_compound': 0,
                    'sentiment_positive': 0,
                    'sentiment_neutral': 1,
                    'sentiment_negative': 0
                }

            scores = self.sentiment_analyzer.polarity_scores(text)
            return {
                'sentiment_compound': scores['compound'],
                'sentiment_positive': scores['pos'],
                'sentiment_neutral': scores['neu'],
                'sentiment_negative': scores['neg']
            }

        sentiment_scores = df['cleaned_text'].apply(get_sentiment_scores)

        df['sentiment_compound'] = sentiment_scores.apply(lambda x: x['sentiment_compound'])
        df['sentiment_positive'] = sentiment_scores.apply(lambda x: x['sentiment_positive'])
        df['sentiment_neutral'] = sentiment_scores.apply(lambda x: x['sentiment_neutral'])
        df['sentiment_negative'] = sentiment_scores.apply(lambda x: x['sentiment_negative'])

        def categorize_sentiment(compound):
            if compound >= 0.05:
                return 'positive'
            if compound <= -0.05:
                return 'negative'
            return 'neutral'

        df['sentiment_category'] = df['sentiment_compound'].apply(categorize_sentiment)

        return df

    def export_for_gis(self, df, filename=None):
        """
        Export a simplified dataset optimized for GIS integration
        For use with ArcGIS StoryMaps as mentioned in the research proposal
        """
        if df.empty:
            return
        filename = filename or self.GIS_FILENAME

        # Add coordinates
        df_gis = df.copy()
        df_gis['latitude'] = df_gis['subreddit'].map(lambda x: CITY_COORDS.get(x, {}).get('lat'))
        df_gis['longitude'] = df_gis['subreddit'].map(lambda x: CITY_COORDS.get(x, {}).get('lon'))
        df_gis['province'] = df_gis['subreddit'].map(lambda x: CITY_COORDS.get(x, {}).get('province'))

        # Select key columns for GIS
        gis_columns = [
            'id', 'subreddit', 'province', 'latitude', 'longitude',
            'created_date', 'year', 'month', 'season', 'quarter',
            *self.GIS_FLAG_COLUMNS,
            'score', 'num_comments', 'engagement_rate'
        ]

        if 'sentiment_compound' in df.columns:
            gis_columns.extend(['sentiment_compound', 'sentiment_category'])

        gis_columns = [col for col in gis_columns if col in df_gis.columns]
        df_gis = df_gis[gis_columns]

        # Save
        df_gis.to_csv(filename, index=False)
        logger.info(f"GIS-ready dataset saved: {filename}")
        logger.info("  Includes lat/lon coordinates for mapping")
        logger.info("  Ready for ArcGIS StoryMaps integration")
