"""
Fraud Scoring Module
Computes composite fraud scores based on multiple features
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def compute_fraud_score(
    behavior_metrics,
    text_similarity_metrics,
    rating_metrics,
    graph_metrics,
    output_dir='outputs',
    weights=None
):
    """
    Compute composite fraud score per user.
    
    Fraud Score Formula:
    fraud_score = 0.3 * behavior_score + 0.25 * similarity_score + 
                  0.2 * rating_entropy_score + 0.25 * cluster_score
    
    Parameters:
    -----------
    behavior_metrics : pd.DataFrame
        Output from compute_behavior_features
    text_similarity_metrics : pd.DataFrame
        Output from compute_text_similarity
    rating_metrics : pd.DataFrame
        Output from compute_rating_entropy
    graph_metrics : pd.DataFrame
        Output from graph_analysis.compute_graph_metrics
    output_dir : str
        Output directory
    weights : dict
        Custom weights for each component. Default: 
        {behavior: 0.3, similarity: 0.25, rating: 0.2, cluster: 0.25}
        
    Returns:
    --------
    pd.DataFrame
        User-level fraud scores with component breakdowns
    """
    
    logger.info("Computing composite fraud scores (optimized)...")
    
    if weights is None:
        weights = {
            'behavior': 0.3,
            'similarity': 0.25,
            'rating': 0.2,
            'cluster': 0.25
        }
    
    logger.info(f"Using weights: {weights}")
    
    # Merge all metrics in one pass
    merged_df = behavior_metrics[['user_id']].copy()
    for df_to_merge in [behavior_metrics, text_similarity_metrics, rating_metrics, graph_metrics]:
        if 'user_id' in df_to_merge.columns and df_to_merge.shape[1] > 1:
            merge_cols = [c for c in df_to_merge.columns if c != 'user_id']
            merged_df = merged_df.merge(
                df_to_merge[['user_id'] + merge_cols], on='user_id', how='left'
            )
    merged_df = merged_df.fillna(0)
    
    # Initialize ONE scaler instance
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # BEHAVIOR SCORE: vectorized
    behavior_features = merged_df[['reviews_per_day', 'burst_activity_score', 'product_concentration']].values.copy()
    behavior_scores = scaler.fit_transform(behavior_features).mean(axis=1)
    
    # SIMILARITY SCORE: vectorized with NaN handling
    if 'avg_text_similarity' in merged_df.columns:
        similarity_features = merged_df[['avg_text_similarity', 'max_text_similarity']].values.copy()
        similarity_features = np.nan_to_num(similarity_features, 0)
        if similarity_features.size > 0 and np.std(similarity_features) > 0:
            similarity_scores = scaler.fit_transform(similarity_features).mean(axis=1)
        else:
            similarity_scores = np.zeros(len(merged_df))
    else:
        similarity_scores = np.zeros(len(merged_df))
    
    # RATING ENTROPY SCORE: vectorized
    rating_features = merged_df[['rating_concentration', 'pct_five_star']].values.copy()
    rating_scores = scaler.fit_transform(rating_features).mean(axis=1)
    
    # CLUSTER/GRAPH SCORE: vectorized with NaN handling
    if 'degree_centrality' in merged_df.columns:
        cluster_features = merged_df[['degree_centrality', 'clustering_coefficient']].values.copy()
        cluster_features = np.nan_to_num(cluster_features, nan=0.0)
        if cluster_features.size > 0 and np.std(cluster_features) > 0:
            cluster_scores = scaler.fit_transform(cluster_features).mean(axis=1)
        else:
            cluster_scores = np.zeros(len(merged_df))
    else:
        cluster_scores = np.zeros(len(merged_df))
    
    # Combine scores with weights
    composite_fraud_score = (
        weights['behavior'] * behavior_scores +
        weights['similarity'] * similarity_scores +
        weights['rating'] * rating_scores +
        weights['cluster'] * cluster_scores
    )
    
    # Create output dataframe
    fraud_scores_df = pd.DataFrame({
        'user_id': merged_df['user_id'],
        'fraud_score': composite_fraud_score,
        'behavior_score': behavior_scores,
        'similarity_score': similarity_scores,
        'rating_entropy_score': rating_scores,
        'cluster_score': cluster_scores,
        'reviews_per_user': merged_df['reviews_per_user'],
        'cluster_id': merged_df['cluster_id'].astype(int),
        'degree': merged_df['degree'].astype(int) if 'degree' in merged_df.columns else 0
    })
    
    # Sort by fraud score
    fraud_scores_df = fraud_scores_df.sort_values('fraud_score', ascending=False).reset_index(drop=True)
    fraud_scores_df['rank'] = range(1, len(fraud_scores_df) + 1)
    
    # Save output
    output_path = Path(output_dir) / 'user_fraud_scores.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fraud_scores_df.to_csv(output_path, index=False)
    
    logger.info(f"Fraud scores computed for {len(fraud_scores_df)} users")
    logger.info(f"Mean fraud score: {fraud_scores_df['fraud_score'].mean():.4f}")
    logger.info(f"Std fraud score: {fraud_scores_df['fraud_score'].std():.4f}")
    logger.info(f"Saved to {output_path}")
    
    # Log top suspicious users
    logger.info("\nTop 10 suspicious users:")
    for idx, row in fraud_scores_df.head(10).iterrows():
        logger.info(f"  {row['user_id']}: {row['fraud_score']:.4f}")
    
    return fraud_scores_df
