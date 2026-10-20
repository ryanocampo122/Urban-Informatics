"""
Reddit Data Collection Pipeline — Pandemic Phase (2020–2021)
-----------------------------------------------------------
Climate Variability, Urban Crime, and Public Perception Study

This script collects, processes, and analyzes Reddit data from Prairie
cities during the COVID-19 pandemic period (2020–2021).

Purpose
-------
- Capture public discourse during a major societal disruption (COVID-19)
- Analyze how pandemic conditions influenced crime and climate discussions
- Generate temporally structured, sentiment-enriched datasets
- Enable comparison with pre-pandemic (2014–2019) and post-pandemic periods

Data Source
-----------
Reddit API via PRAW (Python Reddit API Wrapper)

Outputs
-------
- Monthly intermediate datasets (CSV)
- Final cleaned dataset (CSV)
- Metadata summary (JSON)
- Research summary report (TXT)
- GIS-ready dataset (CSV)
- Collection logs

Methodology
-----------
- Multi-strategy data collection:
    1. Keyword search (crime, weather, pandemic terms)
    2. Top posts (major events)
    3. Hot posts (trending discussions)
    4. New posts (broad coverage)
- Monthly segmentation ensures full temporal coverage
- Expanded keyword sets capture:
    - Crime discourse
    - Weather/climate discourse
    - Pandemic-related discourse
- Feature engineering includes:
    - Temporal indicators (season, quarter, pandemic phase)
    - Content features (keyword counts, overlap flags)
    - Engagement metrics
- Sentiment analysis performed using VADER
- User data anonymized for ethical compliance

Research Context
----------------
- Represents "Pandemic Phase (2020–2021)"
- Captures disruption effects on:
    - crime perception
    - climate awareness
    - online discourse intensity
- Enables analysis of:
    - pre/post pandemic declaration effects
    - vaccine rollout phase
    - variant wave periods
"""
import json
import logging
import os
import sys
import time
from calendar import monthrange
from datetime import datetime, timezone

import pandas as pd

from reddit_common import RedditCollectorBase

# API imports
try:
    import praw
    import prawcore
    PRAW_AVAILABLE = True
except ImportError:
    print("Error: praw not installed. Install with: pip install praw")
    PRAW_AVAILABLE = False
    sys.exit(1)

# Errors raised by PRAW / the Reddit API (network failures, rate limits, 403/404, ...)
API_ERRORS = (praw.exceptions.PRAWException, prawcore.exceptions.PrawcoreException)

# Sentiment analysis
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    print("Warning: vaderSentiment not installed. Install with: pip install vaderSentiment")
    VADER_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('reddit_2020_2021_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Reddit2020To2021Collector(RedditCollectorBase):
    """
    Reddit data collection pipeline for the COVID-19 pandemic period (2020–2021).

    Description
    -----------
    This class manages end-to-end acquisition, filtering, processing,
    feature engineering, and export of Reddit posts related to crime,
    climate, and pandemic discourse.

    Key Responsibilities
    --------------------
    - Collect Reddit data using multiple retrieval strategies
    - Identify and filter crime, weather, and pandemic-related content
    - Generate temporal features aligned with pandemic timeline
    - Compute sentiment metrics using VADER
    - Produce datasets for statistical and geospatial analysis

    Research Role
    -------------
    - Captures behavioral and perceptual shifts during COVID-19
    - Enables comparison with pre-pandemic and post-pandemic periods
    - Supports analysis of compound stressors (climate + crime + pandemic)
    """

    START_YEAR, END_YEAR = 2020, 2021
    GIS_FILENAME = 'reddit_2020_2021_for_gis.csv'
    GIS_FLAG_COLUMNS = (
        'contains_crime_keywords', 'contains_weather_keywords', 'contains_pandemic_keywords',
        'crime_and_weather_and_pandemic',
        'crime_keyword_count', 'weather_keyword_count', 'pandemic_keyword_count',
    )

    def __init__(self, praw_credentials):
        """
        Initialize collector with PRAW credentials

        Args:
            praw_credentials: Dict with keys: client_id, client_secret, user_agent
        """
        if PRAW_AVAILABLE and praw_credentials:
            self.reddit = praw.Reddit(
                client_id=praw_credentials['client_id'],
                client_secret=praw_credentials['client_secret'],
                user_agent=praw_credentials['user_agent']
            )
            logger.info("PRAW initialized for 2020-2021 historical collection")
        else:
            logger.error("PRAW credentials required")
            sys.exit(1)

        # Initialize sentiment analyzer
        if VADER_AVAILABLE:
            self.sentiment_analyzer = SentimentIntensityAnalyzer()
            logger.info("VADER sentiment analyzer initialized")
        else:
            self.sentiment_analyzer = None
            logger.warning("VADER not available - sentiment analysis disabled")

        # Target subreddits - Prairie cities
        self.target_subreddits = [
            'Saskatoon', 'Regina', 'Winnipeg', 'Edmonton',
            'saskatchewan', 'manitoba', 'alberta'
        ]

        # Standard keywords
        self.crime_keywords = [
            'crime', 'theft', 'break-in', 'break in', 'burglary', 'assault',
            'robbery', 'vandalism', 'police', 'cops', 'rcmp', 'safety',
            'dangerous', 'stolen', 'stole', 'carjack', 'mugging', 'murder',
            'shooting', 'stabbing', 'domestic violence', 'gang', 'drug deal',
            'arrest', 'criminal', 'victim', 'attacked', 'security', 'violence',
            'homicide', 'fraud', 'scam', 'harassment', 'break and enter'
        ]

        self.weather_keywords = [
            'cold', 'hot', 'weather', 'temperature', 'snow', 'blizzard',
            'heat wave', 'heatwave', 'winter', 'summer', 'freeze', 'freezing',
            'chinook', 'windchill', 'frost', 'ice storm', 'precipitation',
            'celsius', 'fahrenheit', 'warm', 'humid', 'drought', 'rain',
            'storm', 'climate', 'seasonal', 'spring', 'fall', 'autumn',
            'extreme cold', 'extreme heat', 'tornado', 'flooding', 'wildfire'
        ]

        # New: Pandemic-specific keywords
        self.pandemic_keywords = [
            'covid', 'pandemic', 'virus', 'coronavirus', 'lockdown',
            'quarantine', 'mask', 'vaccine', 'vaccination', 'isolate',
            'outbreak', 'variant', 'delta', 'omicron', 'public health',
            'restriction', 'mandate', 'symptoms', 'testing', 'contact tracing',
            'social distance', 'exposure', 'wave', 'n95', 'pfizer', 'moderna'
        ]

        # Key dates in the COVID-19 Pandemic timeline
        self.pandemic_dates = {
            'pandemic_declared': datetime(2020, 3, 11, tzinfo=timezone.utc),
            'first_wave_peak': datetime(2020, 5, 1, tzinfo=timezone.utc),
            'vaccine_rollout_start': datetime(2020, 12, 1, tzinfo=timezone.utc),
            'delta_variant_wave': datetime(2021, 6, 1, tzinfo=timezone.utc),
            'end_of_2021': datetime(2021, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
        }

    def collect_month_data(self, subreddit, year, month, max_per_search=150):
        """
        Collect posts from a specific month using multiple comprehensive strategies
        Enhanced for historical data collection

        Args:
            subreddit: Subreddit name
            year: Year (2020 or 2021)
            month: Month (1-12)
            max_per_search: Maximum posts per search query

        Returns:
            List of post dictionaries
        """
        month_name = datetime(year, month, 1).strftime('%B %Y')
        logger.info(f"Collecting {month_name} data from r/{subreddit}")

        # Get month boundaries
        first_day = datetime(year, month, 1, tzinfo=timezone.utc)
        last_day_num = monthrange(year, month)[1]
        last_day = datetime(year, month, last_day_num, 23, 59, 59, tzinfo=timezone.utc)

        first_timestamp = int(first_day.timestamp())
        last_timestamp = int(last_day.timestamp())

        posts = []
        seen_ids = set()

        try:
            subreddit_obj = self.reddit.subreddit(subreddit)

            # Strategy 1: Targeted keyword searches (crime + weather + pandemic combined)
            search_terms = [
                'covid', 'lockdown', 'vaccine', 'crime', 'police', 'theft',
                'weather', 'cold', 'snow', 'temperature', 'safety'
            ]

            for term in search_terms:
                try:
                    # Search with time filter
                    for submission in subreddit_obj.search(
                        term,
                        time_filter='all',
                        limit=max_per_search,
                        sort='relevance'
                    ):
                        post_timestamp = int(submission.created_utc)

                        if first_timestamp <= post_timestamp <= last_timestamp:
                            if submission.id not in seen_ids:
                                posts.append(self._extract_post_data(submission))
                                seen_ids.add(submission.id)

                    time.sleep(2.5)  # Conservative rate limiting

                except API_ERRORS as e:
                    logger.warning(f"Search failed for '{term}': {str(e)}")
                    time.sleep(5)
                    continue

            # Strategy 2: Top posts from year (captures major events)
            try:
                for submission in subreddit_obj.top(time_filter='year', limit=500):
                    post_timestamp = int(submission.created_utc)

                    if first_timestamp <= post_timestamp <= last_timestamp:
                        if submission.id not in seen_ids:
                            posts.append(self._extract_post_data(submission))
                            seen_ids.add(submission.id)

                time.sleep(2)
            except API_ERRORS as e:
                logger.warning(f"Top posts collection failed: {str(e)}")

            # Strategy 3: Hot posts (captures what was trending)
            try:
                for submission in subreddit_obj.hot(limit=200):
                    post_timestamp = int(submission.created_utc)

                    if first_timestamp <= post_timestamp <= last_timestamp:
                        if submission.id not in seen_ids:
                            posts.append(self._extract_post_data(submission))
                            seen_ids.add(submission.id)

                time.sleep(2)
            except API_ERRORS as e:
                logger.warning(f"Hot posts collection failed: {str(e)}")

            # Strategy 4: New posts (good for complete coverage)
            try:
                for submission in subreddit_obj.new(limit=500):
                    post_timestamp = int(submission.created_utc)

                    if first_timestamp <= post_timestamp <= last_timestamp:
                        if submission.id not in seen_ids:
                            posts.append(self._extract_post_data(submission))
                            seen_ids.add(submission.id)

                time.sleep(2)
            except API_ERRORS as e:
                logger.warning(f"New posts collection failed: {str(e)}")

            logger.info(f"  Collected {len(posts)} unique posts from {month_name}")

        except API_ERRORS as e:
            logger.error(f"Error collecting from r/{subreddit} for {month_name}: {str(e)}")

        return posts

    def filter_relevant_posts(self, posts):
        """Filter posts for crime, weather, OR pandemic keywords"""
        relevant_posts = []

        for post in posts:
            full_text = f"{post.get('title', '')} {post.get('selftext', '')}".lower()

            # Check for crime keywords
            has_crime = any(keyword in full_text for keyword in self.crime_keywords)

            # Check for weather keywords
            has_weather = any(keyword in full_text for keyword in self.weather_keywords)

            # Check for pandemic keywords
            has_pandemic = any(keyword in full_text for keyword in self.pandemic_keywords)

            if has_crime or has_weather or has_pandemic:
                relevant_posts.append(post)

        return relevant_posts

    def process_posts(self, posts, subreddit):
        """Process raw posts into structured data"""
        if not posts:
            return pd.DataFrame()

        df = pd.DataFrame(posts)
        df['subreddit'] = subreddit

        df = self._clean_data(df)
        df = self._add_temporal_features(df)
        df = self._add_content_features(df)
        df = self._add_pandemic_flags(df)  # <-- Updated
        df = self._anonymize_data(df)

        if self.sentiment_analyzer:
            df = self._add_sentiment(df)

        return df

    def _add_temporal_features(self, df):
        """Add temporal features for analysis"""
        df['year'] = df['created_date'].dt.year
        df['month'] = df['created_date'].dt.month
        df['day'] = df['created_date'].dt.day
        df['day_of_week'] = df['created_date'].dt.dayofweek
        df['day_name'] = df['created_date'].dt.day_name()
        df['week_of_year'] = df['created_date'].dt.isocalendar().week
        df['hour'] = df['created_date'].dt.hour

        # Season classification
        def get_season(month):
            if month in [12, 1, 2]:
                return 'winter'
            if month in [3, 4, 5]:
                return 'spring'
            if month in [6, 7, 8]:
                return 'summer'
            return 'fall'

        df['season'] = df['month'].apply(get_season)

        # Platform period (all posts in this dataset are in API changes era)
        df['platform_period'] = '2020-2021_pandemic_era'  # <-- Updated

        return df

    def _add_pandemic_flags(self, df):
        """
        Add COVID-19 timeline features.

        Description
        -----------
        Annotates posts based on key pandemic milestones to enable
        phase-based temporal analysis.

        Features
        --------
        - Pre vs post WHO pandemic declaration (March 11, 2020)
        - Vaccine rollout phase (December 2020 onward)
        - Major variant wave period (mid-2021 onward)
        - Quarterly segmentation across 2020–2021

        Research Value
        --------------
        Enables:
        - Before/after analysis of discourse shifts
        - Pandemic-phase comparisons
        - Integration with external datasets (policy, restrictions, etc.)
        """

        df['is_2020'] = df['year'] == 2020
        df['is_2021'] = df['year'] == 2021

        # Flag posts relative to key pandemic dates
        df['pre_pandemic_declared'] = df['created_date'] < self.pandemic_dates['pandemic_declared']
        df['post_pandemic_declared'] = df['created_date'] >= self.pandemic_dates['pandemic_declared']

        # During vaccine rollout period
        df['vaccine_rollout'] = (
            df['created_date'] >= self.pandemic_dates['vaccine_rollout_start']
        )

        # During major 2021 variant wave (Delta/Omicron)
        df['major_variant_wave'] = (
            df['created_date'] >= self.pandemic_dates['delta_variant_wave']
        )

        # Quarterly breakdown for 2020-2021
        def get_quarter(row):
            year = 2020 if row['year'] == 2020 else 2021
            return f"{year}_Q{(row['month'] - 1) // 3 + 1}"

        df['quarter'] = df.apply(get_quarter, axis=1)

        return df

    def _add_content_features(self, df):
        """Add content-based features"""
        # Clean text
        df['cleaned_text'] = df['full_text'].apply(self._clean_text)

        # Text length metrics
        df['text_length'] = df['cleaned_text'].str.len()
        df['word_count'] = df['cleaned_text'].str.split().str.len()

        # Topic flags
        crime_pattern = '|'.join(self.crime_keywords)
        weather_pattern = '|'.join(self.weather_keywords)
        pandemic_pattern = '|'.join(self.pandemic_keywords)  # <-- New

        df['contains_crime_keywords'] = df['full_text'].str.contains(
            crime_pattern, case=False, na=False, regex=True
        )
        df['contains_weather_keywords'] = df['full_text'].str.contains(
            weather_pattern, case=False, na=False, regex=True
        )
        df['contains_pandemic_keywords'] = df['full_text'].str.contains(  # <-- New
            pandemic_pattern, case=False, na=False, regex=True
        )

        # Count keywords
        df['crime_keyword_count'] = df['full_text'].str.lower().apply(
            lambda x: sum(keyword in x for keyword in self.crime_keywords)
        )
        df['weather_keyword_count'] = df['full_text'].str.lower().apply(
            lambda x: sum(keyword in x for keyword in self.weather_keywords)
        )
        df['pandemic_keyword_count'] = df['full_text'].str.lower().apply(  # <-- New
            lambda x: sum(keyword in x for keyword in self.pandemic_keywords)
        )

        # Combination flag (Crime AND Weather AND Pandemic)
        df['crime_and_weather_and_pandemic'] = (  # <-- Updated
            df['contains_crime_keywords'] &
            df['contains_weather_keywords'] &
            df['contains_pandemic_keywords']
        )

        # Engagement metrics
        df['engagement_rate'] = df['num_comments'] / (df['score'] + 1)
        df['controversy_score'] = 1 - df['upvote_ratio']

        return df

    def collect_2020_2021_data(self, save_intermediate=True):  # <-- Updated method name
        """
        Execute full data collection pipeline for pandemic period.

        Description
        -----------
        Iterates through all months (2020–2021), collecting and processing
        Reddit data across all target subreddits.

        Parameters
        ----------
        save_intermediate : bool
            Whether to save monthly datasets for backup and validation.

        Returns
        -------
        pandas.DataFrame
            Combined dataset across all cities and months.

        Notes
        -----
        - Covers 24 months of pandemic-era data
        - Applies filtering, cleaning, and feature engineering
        - Outputs analysis-ready dataset for integration with:
        - climate data (ECCC)
        - crime statistics (municipal datasets)
        """
        logger.info("="*70)
        logger.info("Starting 2020-2021 Historical Data Collection")
        logger.info("Target Period: COVID-19 Pandemic Era")
        logger.info("="*70)

        # Generate all months in 2020-2021
        months_to_collect = self.generate_month_list(
            start_year=2020, start_month=1,
            end_year=2021, end_month=12
        )

        all_data = []

        for year, month in months_to_collect:
            month_name = datetime(year, month, 1).strftime('%Y-%m')

            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {month_name}")
            logger.info(f"{'='*60}")

            month_posts = []

            for subreddit in self.target_subreddits:
                try:
                    # Collect posts for this month
                    posts = self.collect_month_data(subreddit, year, month)

                    if posts:
                        # Filter for relevant content
                        relevant_posts = self.filter_relevant_posts(posts)
                        logger.info(f"  r/{subreddit}: {len(posts)} collected, "
                                    f"{len(relevant_posts)} relevant")

                        # Process posts
                        if relevant_posts:
                            df = self.process_posts(relevant_posts, subreddit)
                            month_posts.append(df)

                    time.sleep(3)  # Conservative rate limiting

                except API_ERRORS + (KeyError, ValueError) as e:
                    logger.error(f"Error processing r/{subreddit}: {str(e)}")
                    time.sleep(5)
                    continue

            # Combine month data
            if month_posts:
                month_df = pd.concat(month_posts, ignore_index=True)
                month_df = month_df.drop_duplicates(subset=['id'])

                all_data.append(month_df)

                if save_intermediate:
                    filename = f"reddit_2020_2021_{month_name}.csv"  # <-- Updated
                    month_df.to_csv(filename, index=False)
                    logger.info(f"Saved: {filename} ({len(month_df)} posts)")

        # Combine all data
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['id'])
            combined_df = combined_df.sort_values('created_date')

            logger.info(f"\n{'='*60}")
            logger.info("COLLECTION COMPLETE - 2020-2021 DATASET")  # <-- Updated
            logger.info(f"{'='*60}")
            logger.info(f"Total posts collected: {len(combined_df)}")
            logger.info(f"Date range: {combined_df['created_date'].min()} to "
                        f"{combined_df['created_date'].max()}")
            logger.info(f"Months covered: {len(months_to_collect)}")

            return combined_df
        logger.warning("No data collected")
        return pd.DataFrame()

    def _print_summary(self, df):
        """Print comprehensive summary statistics"""
        logger.info("\n%s", "=" * 70)
        logger.info("2020-2021 PANDEMIC DATASET SUMMARY")  # <-- Updated
        logger.info("="*70)

        logger.info(f"\nTotal Posts: {len(df)}")
        logger.info(f"Date Range: {df['created_date'].min()} to {df['created_date'].max()}")

        logger.info("\n--- Posts by Subreddit ---")
        for sub, count in df['subreddit'].value_counts().items():
            logger.info(f"  r/{sub}: {count} ({count/len(df)*100:.1f}%)")

        logger.info("\n--- Posts by Year ---")
        for year, count in df['year'].value_counts().sort_index().items():
            logger.info(f"  {year}: {count}")

        logger.info("\n--- Posts by Quarter ---")
        for quarter, count in df['quarter'].value_counts().sort_index().items():
            logger.info(f"  {quarter}: {count}")

        logger.info("\n--- Posts by Season ---")
        for season, count in df['season'].value_counts().items():
            logger.info(f"  {season}: {count}")

        logger.info("\n--- Content Analysis ---")
        logger.info(f"Crime-related posts: {df['contains_crime_keywords'].sum()} "
                    f"({df['contains_crime_keywords'].mean()*100:.1f}%)")
        logger.info(f"Weather-related posts: {df['contains_weather_keywords'].sum()} "
                    f"({df['contains_weather_keywords'].mean()*100:.1f}%)")
        logger.info(f"Pandemic-related posts: {df['contains_pandemic_keywords'].sum()} "  # <-- New
                    f"({df['contains_pandemic_keywords'].mean()*100:.1f}%)")
        logger.info(f"Crime AND Weather AND Pandemic: "  # <-- Updated
                    f"{df['crime_and_weather_and_pandemic'].sum()} "
                    f"({df['crime_and_weather_and_pandemic'].mean()*100:.1f}%)")

        logger.info("\n--- Pandemic Timeline ---")  # <-- Updated
        logger.info(f"Pre-pandemic declared (before Mar 11, 2020): {df['pre_pandemic_declared'].sum()}")
        logger.info(f"Post-pandemic declared: {df['post_pandemic_declared'].sum()}")
        logger.info(f"During vaccine rollout (Dec 2020+): {df['vaccine_rollout'].sum()}")
        logger.info(f"During major 2021 variant wave (Jun 2021+): {df['major_variant_wave'].sum()}")

        # Add sentiment summary if present
        if 'sentiment_category' in df.columns:
            logger.info("\n--- Sentiment Summary ---")
            sentiment_counts = df['sentiment_category'].value_counts()
            for cat, count in sentiment_counts.items():
                logger.info(f"  {cat}: {count} ({count/len(df)*100:.1f}%)")
            logger.info(f"Average compound sentiment: {df['sentiment_compound'].mean():.3f}")

        # Engagement metrics
        logger.info("\n--- Engagement Statistics ---")
        logger.info(f"Average score: {df['score'].mean():.1f}")
        logger.info(f"Average comments: {df['num_comments'].mean():.1f}")
        logger.info(f"Average upvote ratio: {df['upvote_ratio'].mean():.3f}")
        logger.info(f"Average engagement rate: {df['engagement_rate'].mean():.3f}")
        logger.info(f"Average controversy score: {df['controversy_score'].mean():.3f}")

        logger.info("\n%s", "=" * 70)
        logger.info("SUMMARY COMPLETE")
        logger.info("="*70)

    def save_final_dataset(self, df, filename='reddit_climate_crime_pandemic_2020_2021.csv'):  # <-- Updated
        """Save final analysis-ready dataset and metadata"""
        if df.empty:
            logger.error("Cannot save empty dataset")
            return

        # Select columns for final dataset
        columns_to_keep = [
            'id', 'subreddit', 'author_hash', 'created_date', 'created_utc',
            'year', 'month', 'day', 'day_of_week', 'day_name', 'week_of_year',
            'hour', 'season', 'quarter', 'platform_period',
            'is_2020', 'is_2021', 'pre_pandemic_declared', 'post_pandemic_declared',  # <-- Updated
            'vaccine_rollout', 'major_variant_wave',  # <-- Updated
            'title', 'cleaned_text', 'text_length', 'word_count',
            'link_flair_text', 'contains_crime_keywords', 'contains_weather_keywords',
            'contains_pandemic_keywords', 'crime_and_weather_and_pandemic',  # <-- Updated
            'crime_keyword_count', 'weather_keyword_count', 'pandemic_keyword_count',  # <-- Updated
            'score', 'upvote_ratio', 'num_comments', 'engagement_rate',
            'controversy_score', 'over_18', 'spoiler', 'stickied', 'locked'
        ]

        if 'sentiment_compound' in df.columns:
            columns_to_keep.extend([
                'sentiment_compound', 'sentiment_positive',
                'sentiment_neutral', 'sentiment_negative', 'sentiment_category'
            ])

        columns_to_keep = [col for col in columns_to_keep if col in df.columns]
        final_df = df[columns_to_keep].copy()

        # Save to CSV
        final_df.to_csv(filename, index=False)
        logger.info(f"\nFinal dataset saved: {filename}")
        logger.info(f"Shape: {final_df.shape}")

        # Save metadata
        metadata = {
            'collection_date': datetime.now().isoformat(),
            'time_period': '2020-2021 (COVID-19 Pandemic Era)',  # <-- Updated
            'total_posts': len(final_df),
            'date_range': {
                'start': str(final_df['created_date'].min()),
                'end': str(final_df['created_date'].max())
            },
            'subreddits': final_df['subreddit'].value_counts().to_dict(),
            'posts_by_year': final_df['year'].value_counts().sort_index().to_dict(),
            'posts_by_quarter': final_df['quarter'].value_counts().sort_index().to_dict(),
            'crime_related': int(final_df['contains_crime_keywords'].sum()),
            'weather_related': int(final_df['contains_weather_keywords'].sum()),
            'pandemic_related': int(final_df['contains_pandemic_keywords'].sum()),  # <-- New
            'crime_and_weather_and_pandemic': int(
                final_df['crime_and_weather_and_pandemic'].sum()),  # <-- Updated
            'pandemic_timeline_breakdown': {  # <-- Updated
                'pre_declared': int(final_df['pre_pandemic_declared'].sum()),
                'post_declared': int(final_df['post_pandemic_declared'].sum()),
                'vaccine_rollout': int(final_df['vaccine_rollout'].sum()),
                'major_variant_wave': int(final_df['major_variant_wave'].sum())
            }
        }

        metadata_filename = filename.replace('.csv', '_metadata.json')
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)

        logger.info(f"Metadata saved: {metadata_filename}")

        # Print summary
        self._print_summary(final_df)

    def generate_research_summary(self, df,
                                  filename='research_questions_3_4_pandemic_summary.txt'):  # <-- Updated
        """Generate a focused summary for research questions 3 and 4 in the pandemic context"""
        if df.empty:
            return

        with open(filename, 'w', encoding='utf-8') as f:
            f.write("="*70 + "\n")
            f.write("RESEARCH QUESTIONS 3 & 4 - ANALYSIS SUMMARY\n")
            f.write("2020-2021 PANDEMIC REDDIT DATA COLLECTION\n")  # <-- Updated
            f.write("="*70 + "\n\n")

            f.write("QUESTION 3: How did public discussions of crime and climate on social media\n")
            f.write("(Reddit) change *during* the initial phases of the COVID-19 pandemic?\n")  # <-- Updated
            f.write("-"*70 + "\n\n")

            # Sentiment by season
            f.write("SENTIMENT BY SEASON (OVERALL):\n")
            for season in ['winter', 'spring', 'summer', 'fall']:
                season_data = df[df['season'] == season]
                if len(season_data) > 0 and 'sentiment_compound' in df.columns:
                    avg_sentiment = season_data['sentiment_compound'].mean()
                    f.write(f"  {season.capitalize()}: {avg_sentiment:.3f} "
                            f"(n={len(season_data)} posts)\n")

            f.write("\nCRIME DISCUSSIONS BY PANDEMIC PHASE:\n")  # <-- New Focus
            f.write(f"  Total crime posts: {df['contains_crime_keywords'].sum()}\n")

            f.write(f"  Pre-Pandemic Declared: "
                    f"{df[df['pre_pandemic_declared']]['contains_crime_keywords'].sum()} \n")
            f.write(f"  Post-Pandemic Declared: "
                    f"{df[df['post_pandemic_declared']]['contains_crime_keywords'].sum()} \n")
            f.write(f"  Vaccine Rollout (Dec 2020+): "
                    f"{df[df['vaccine_rollout']]['contains_crime_keywords'].sum()} \n")

            f.write("\nWEATHER DISCUSSIONS BY PANDEMIC PHASE:\n")  # <-- New Focus
            f.write(f"  Total weather posts: {df['contains_weather_keywords'].sum()}\n")

            f.write(f"  Pre-Pandemic Declared: "
                    f"{df[df['pre_pandemic_declared']]['contains_weather_keywords'].sum()} \n")
            f.write(f"  Post-Pandemic Declared: "
                    f"{df[df['post_pandemic_declared']]['contains_weather_keywords'].sum()} \n")
            f.write(f"  Vaccine Rollout (Dec 2020+): "
                    f"{df[df['vaccine_rollout']]['contains_weather_keywords'].sum()} \n")

            f.write("\n" + "="*70 + "\n\n")
            f.write("QUESTION 4: What spatial patterns emerge when mapping climate trends, crime,\n")
            f.write("and the prevalence/sentiment of *pandemic discussion* "
                    "in Prairie cities?\n")  # <-- Updated
            f.write("-"*70 + "\n\n")

            # Posts by city
            f.write("POSTS BY CITY/REGION:\n")
            for sub in df['subreddit'].value_counts().index:
                sub_data = df[df['subreddit'] == sub]
                crime_pct = sub_data['contains_crime_keywords'].mean() * 100
                weather_pct = sub_data['contains_weather_keywords'].mean() * 100
                pandemic_pct = sub_data['contains_pandemic_keywords'].mean() * 100  # <-- New

                if 'sentiment_compound' in df.columns:
                    avg_sentiment = sub_data['sentiment_compound'].mean()
                    f.write(f"  r/{sub}: {len(sub_data)} posts\n")
                    f.write(f"    - Crime mentions: {crime_pct:.1f}%\n")
                    f.write(f"    - Weather mentions: {weather_pct:.1f}%\n")
                    f.write(f"    - **Pandemic mentions**: {pandemic_pct:.1f}%\n")  # <-- New
                    f.write(f"    - Avg sentiment: {avg_sentiment:.3f}\n")
                else:
                    f.write(f"  r/{sub}: {len(sub_data)} posts\n")
                    f.write(f"    - Crime mentions: {crime_pct:.1f}%\n")
                    f.write(f"    - Weather mentions: {weather_pct:.1f}%\n")
                    f.write(f"    - **Pandemic mentions**: {pandemic_pct:.1f}%\n")

            f.write("\nCRIME-WEATHER-PANDEMIC CORRELATION BY CITY:\n")  # <-- Updated
            for sub in df['subreddit'].value_counts().index:
                sub_data = df[df['subreddit'] == sub]
                triple_overlap = sub_data['crime_and_weather_and_pandemic'].sum()
                if len(sub_data) > 0:
                    pct = triple_overlap / len(sub_data) * 100
                    f.write(f"  r/{sub}: {triple_overlap} posts mention all three ({pct:.1f}%)\n")

            f.write("\n" + "="*70 + "\n\n")
            f.write("TEMPORAL ANALYSIS:\n")
            f.write("-"*70 + "\n\n")

            f.write("POSTS BY QUARTER:\n")
            for quarter in sorted(df['quarter'].unique()):
                quarter_data = df[df['quarter'] == quarter]
                f.write(f"  {quarter}: {len(quarter_data)} posts\n")

            f.write("\nPANDEMIC PHASE POST COUNTS:\n")  # <-- Updated
            f.write(f"  Pre-Pandemic declared: {df['pre_pandemic_declared'].sum()} posts\n")
            f.write(f"  Post-Pandemic declared: {df['post_pandemic_declared'].sum()} posts\n")
            f.write(f"  Vaccine Rollout (Dec 2020+): {df['vaccine_rollout'].sum()} posts\n")
            f.write(f"  Major 2021 Variant Wave (Jun 2021+): {df['major_variant_wave'].sum()} posts\n")

            f.write("\n" + "="*70 + "\n")
            f.write("DATA READY FOR:\n")
            f.write("1. Correlation analysis with climate data (Environment Canada)\n")
            f.write("2. Correlation analysis with crime data (municipal databases)\n")
            f.write("3. Geospatial mapping of combined discussion (ArcGIS StoryMaps)\n")
            f.write("4. Time-series analysis of crime/weather sentiment vs. pandemic phases\n")
            f.write("5. Statistical testing (regression, ANOVA, etc.)\n")
            f.write("="*70 + "\n")

        logger.info(f"Research summary saved: {filename}")


def main():
    """Main execution function for 2020-2021 data collection"""
    logger.info("=" * 70)
    logger.info("Reddit 2020-2021 Historical Data Collection")  # <-- Updated
    logger.info("Prairie Cities Climate-Crime-Pandemic Research")  # <-- Updated
    logger.info("Focus: COVID-19 Pandemic Era")  # <-- Updated
    logger.info("=" * 70)

    # ── Reddit API Credentials ─────────────────────────────────────────────
    # Set credentials via environment variables (recommended for sharing):
    #
    #   export REDDIT_CLIENT_ID="your_client_id_here"
    #   export REDDIT_CLIENT_SECRET="your_client_secret_here"
    #   export REDDIT_USER_AGENT="your_app_name/1.0 (by u/your_username)"
    #
    # Alternatively, create a .env file in the same directory:
    #   REDDIT_CLIENT_ID=your_client_id_here
    #   REDDIT_CLIENT_SECRET=your_client_secret_here
    #   REDDIT_USER_AGENT=prairie-urban-safety/1.0 (by u/your_username)
    #
    # How to get credentials:
    #   1. Go to https://www.reddit.com/prefs/apps
    #   2. Click "Create another app" at the bottom
    #   3. Select "script", give it any name, set redirect URI to http://localhost:8080
    #   4. Your client_id is the string under the app name
    #   5. Your client_secret is labelled "secret"
    #
    # Never commit credentials to a public repository.
    # ────────────────────────────────────────────────────────────────────────

    # Load from environment (try python-dotenv if available)
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv not required — use export commands instead

    client_id = os.environ.get('REDDIT_CLIENT_ID')
    client_secret = os.environ.get('REDDIT_CLIENT_SECRET')
    user_agent = os.environ.get('REDDIT_USER_AGENT', 'prairie-urban-safety/1.0')

    if not client_id or not client_secret:
        logger.error("Reddit API credentials not found in environment variables.")
        logger.error("Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET before running.")
        logger.error("See script header for instructions.")
        sys.exit(1)

    praw_credentials = {
        'client_id':     client_id,
        'client_secret': client_secret,
        'user_agent':    user_agent,
    }

    output_filename = 'reddit_climate_crime_pandemic_2020_2021.csv'  # <-- Updated

    logger.info("\nInitializing collector...")
    collector = Reddit2020To2021Collector(praw_credentials=praw_credentials)  # <-- Updated Class

    logger.info("\nStarting data collection for 2020-2021 period...")  # <-- Updated
    logger.info("This will collect 24 months of historical data")
    logger.info("Estimated time: 30-60 minutes depending on data volume\n")

    # Collect all data
    df = collector.collect_2020_2021_data(save_intermediate=True)  # <-- Updated method call

    if not df.empty:
        # Save final dataset
        collector.save_final_dataset(df, filename=output_filename)

        # Generate research-focused summary
        collector.generate_research_summary(df)

        # Export GIS-ready data
        collector.export_for_gis(df)

        logger.info("\n%s", "=" * 70)
        logger.info("COLLECTION SUCCESSFUL")
        logger.info("="*70)
        logger.info(f"Main dataset: {output_filename}")
        logger.info(f"Metadata: {output_filename.replace('.csv', '_metadata.json')}")
        logger.info("Research summary: research_questions_3_4_pandemic_summary.txt")
        logger.info("GIS dataset: reddit_2020_2021_for_gis.csv")
        logger.info("Individual monthly files: reddit_2020_2021_YYYY-MM.csv")
        logger.info("Log file: reddit_2020_2021_scraper.log")

        # Research-specific insights
        logger.info("\n%s", "=" * 70)
        logger.info("RESEARCH QUESTIONS 3 & 4 - KEY INSIGHTS (Pandemic Era)")  # <-- Updated
        logger.info("="*70)

        logger.info("\nQuestion 3: Public perceptions and discussions variation during the pandemic")
        logger.info("Dataset includes temporal markers for:")
        logger.info("- Seasonal variations (winter/spring/summer/fall)")
        logger.info("- Pandemic timeline "
                    "(pre/post declared, vaccine rollout, major variant wave)")  # <-- Updated
        logger.info("- Sentiment analysis for each post")
        logger.info("- Day of week, hour, and quarterly breakdowns")

        logger.info("\nQuestion 4: Spatial patterns")
        logger.info("Dataset covers 4 cities + 3 provincial subreddits:")
        for city in collector.target_subreddits:
            city_count = len(df[df['subreddit'] == city])
            logger.info(f"- r/{city}: {city_count} posts")

        logger.info("\nNext steps for analysis:")
        logger.info("1. Merge with climate data (Environment Canada)")
        logger.info("2. Merge with crime data (municipal police services)")
        logger.info("3. Perform temporal correlation analysis (e.g., crime vs. lockdown)")  # <-- Updated
        logger.info("4. Create spatial visualizations (GIS/StoryMaps)")
        logger.info("5. Analyze sentiment variations by season/weather and pandemic phase")  # <-- Updated
        logger.info("6. Statistical testing (regression, time-series)")

        logger.info("\n%s", "=" * 70)
        logger.info("FILES GENERATED:")
        logger.info("="*70)
        logger.info(f"✓ {output_filename} - Main analysis dataset")
        logger.info(f"✓ {output_filename.replace('.csv', '_metadata.json')} - Collection metadata")
        logger.info("✓ research_questions_3_4_pandemic_summary.txt - Focused research summary")
        logger.info("✓ reddit_2020_2021_for_gis.csv - GIS-ready with coordinates")
        logger.info("✓ reddit_2020_2021_YYYY-MM.csv - Monthly backup files (24 files)")
        logger.info("✓ reddit_2020_2021_scraper.log - Detailed collection log")

    else:
        logger.error("\n%s", "=" * 70)
        logger.error("COLLECTION FAILED - No data collected")
        logger.error("="*70)
        logger.error("Please check:")
        logger.error("1. PRAW credentials are valid")
        logger.error("2. Internet connection is stable")
        logger.error("3. Reddit API is accessible")
        logger.error("4. Rate limits have not been exceeded")
        logger.error("\nCheck reddit_2020_2021_scraper.log for details")


if __name__ == "__main__":
    if not PRAW_AVAILABLE:
        print("Error: PRAW is required. Install with: pip install praw")
        sys.exit(1)

    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\nCollection interrupted by user")
        logger.info("Partial data may have been saved to monthly files")
    except Exception as e:
        logger.error(f"\nFatal error: {str(e)}", exc_info=True)
        logger.info("\nTroubleshooting tips:")
        logger.info("1. Check reddit_2020_2021_scraper.log for detailed errors")
        logger.info("2. Verify PRAW credentials")
        logger.info("3. Check if any monthly files were created")
        logger.info("4. Try reducing max_per_search parameter if rate limited")
        raise SystemExit(1) from e
