"""
Behavioral Features Module
Computes user-level behavioral metrics indicating suspicious activity
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def compute_behavior_features(df, output_dir='outputs'):
    """
    Compute behavioral features for fraud detection (OPTIMIZED - vectorized).
    
    Features computed:
    - reviews_per_user: Total number of reviews
    - reviews_per_day: Average reviews per day
    - burst_activity_score: Concentration of reviews in time windows
    - unique_products_per_user: Diversity of products reviewed
    - avg_time_between_reviews: Time patterns
    """
    
    logger.info("Computing behavioral features (vectorized)...")
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Basic metrics (already efficient)
    reviews_per_user = df.groupby('user_id').size()
    time_span = df.groupby('user_id')['timestamp'].agg(['min', 'max'])
    time_span['days_active'] = (time_span['max'] - time_span['min']).dt.days + 1
    time_span.loc[time_span['days_active'] == 0, 'days_active'] = 1
    reviews_per_day = reviews_per_user / time_span['days_active']
    unique_products = df.groupby('user_id')['product_id'].nunique()
    product_concentration = (1 - (unique_products / reviews_per_user)).fillna(0)
    
    # Vectorized burst score (daily review variance per user)
    df_temp = df.copy()
    df_temp['date'] = df_temp['timestamp'].dt.date
    daily_counts = df_temp.groupby(['user_id', 'date']).size().reset_index(name='count')
    daily_stats = daily_counts.groupby('user_id')['count'].agg(['std', 'mean', 'count']).fillna(0)
    burst_scores = daily_stats['std'] / (daily_stats['mean'] + 1e-6)
    burst_scores[daily_stats['count'] <= 1] = 0
    
    # Vectorized time between reviews
    df_sorted = df.sort_values(['user_id', 'timestamp']).copy()
    df_sorted['time_diff'] = df_sorted.groupby('user_id')['timestamp'].diff().dt.total_seconds() / 3600
    avg_time_between = df_sorted.groupby('user_id')['time_diff'].mean().fillna(0)
    
    # Combine all features
    behavior_metrics = pd.DataFrame({
        'user_id': reviews_per_user.index,
        'reviews_per_user': reviews_per_user.values,
        'reviews_per_day': reviews_per_day.values,
        'burst_activity_score': burst_scores.loc[reviews_per_user.index].values,
        'unique_products': unique_products.values,
        'product_concentration': product_concentration.values,
        'avg_hours_between_reviews': avg_time_between.loc[reviews_per_user.index].values,
        'days_active': time_span['days_active'].loc[reviews_per_user.index].values
    })
    
    behavior_metrics = behavior_metrics.fillna(0)
    
    # Save output
    output_path = Path(output_dir) / 'user_behavior_metrics.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    behavior_metrics.to_csv(output_path, index=False)
    logger.info(f"Behavioral metrics saved to {output_path}")
    logger.info(f"Computed metrics for {len(behavior_metrics)} users")
    
    return behavior_metrics
