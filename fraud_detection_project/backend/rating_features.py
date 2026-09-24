"""
Rating Features Module
Computes entropy-based features from rating distributions
"""

import pandas as pd
import numpy as np
from scipy.stats import entropy
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def compute_rating_entropy(df, output_dir='outputs'):
    """
    Compute rating entropy features per user.
    
    High entropy = diverse ratings (legitimate)
    Low entropy = concentrated ratings (suspicious - e.g., always 5 stars)
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset with columns: user_id, rating
    output_dir : str
        Output directory to save metrics
        
    Returns:
    --------
    pd.DataFrame
        User-level rating metrics
    """
    
    logger.info("Computing rating entropy features (vectorized)...")
    
    # Ensure rating is numeric
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])
    
    # Vectorized basic stats
    rating_agg = df.groupby('user_id')['rating'].agg(['mean', 'std', 'count']).fillna(0)
    
    # Vectorized percentages
    five_star_pct = (df['rating'] == 5).groupby(df['user_id']).sum() / df.groupby('user_id').size()
    positive_pct = (df['rating'] >= 4).groupby(df['user_id']).sum() / df.groupby('user_id').size()
    
    # Vectorized entropy calculation
    max_entropy = np.log(5)
    rating_entropies = []
    
    for user_id, group_df in df.groupby('user_id'):
        counts = group_df['rating'].value_counts()
        probs = counts.values / counts.sum()
        user_entropy = entropy(probs) / max_entropy if max_entropy > 0 else 0
        rating_entropies.append([user_id, user_entropy])
    
    entropy_df = pd.DataFrame(rating_entropies, columns=['user_id', 'rating_entropy']).set_index('user_id')
    
    # Combine all metrics
    rating_df = pd.DataFrame({
        'user_id': rating_agg.index,
        'rating_entropy': entropy_df.loc[rating_agg.index, 'rating_entropy'].values,
        'avg_rating': rating_agg['mean'].values,
        'std_rating': rating_agg['std'].values,
        'rating_concentration': 1 - entropy_df.loc[rating_agg.index, 'rating_entropy'].values,
        'pct_five_star': five_star_pct.loc[rating_agg.index].fillna(0).values,
        'pct_positive_ratings': positive_pct.loc[rating_agg.index].fillna(0).values,
        'num_ratings': rating_agg['count'].values.astype(int)
    })
    
    rating_df = rating_df.fillna(0)
    
    # Save output
    output_path = Path(output_dir) / 'user_rating_metrics.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rating_df.to_csv(output_path, index=False)
    logger.info(f"Rating entropy metrics saved to {output_path}")
    logger.info(f"Computed rating metrics for {len(rating_df)} users")
    
    return rating_df
