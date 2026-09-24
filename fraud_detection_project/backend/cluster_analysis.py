"""
Cluster Analysis Module
Maps clusters to targeted products and analyzes cluster characteristics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def map_cluster_to_products(
    df,
    cluster_membership,
    fraud_scores,
    output_dir='outputs',
    fraud_threshold=0.5
):
    """
    Map each cluster to its targeted products.
    
    For each cluster, identify:
    - All products reviewed by cluster members
    - Average fraud score in cluster
    - Concentration of reviews on specific products
    
    Parameters:
    -----------
    df : pd.DataFrame
        Original dataset with columns: user_id, product_id, rating
    cluster_membership : pd.DataFrame
        Cluster assignments (from build_user_graph)
    fraud_scores : pd.DataFrame
        Fraud scores per user
    output_dir : str
        Output directory
    fraud_threshold : float
        Threshold for flagging cluster as suspicious
        
    Returns:
    --------
    pd.DataFrame
        Cluster-product relationships
    """
    
    logger.info("Mapping clusters to targeted products (vectorized)...")
    
    # Merge cluster info with fraud scores
    cluster_fraud = cluster_membership.merge(
        fraud_scores[['user_id', 'fraud_score']],
        on='user_id',
        how='left'
    ).fillna(0)
    
    # Vectorized cluster-product mapping
    # Precompute cluster stats
    cluster_stats = cluster_fraud.groupby('cluster_id').agg({
        'fraud_score': 'mean',
        'user_id': 'nunique'
    })
    cluster_stats.columns = ['avg_fraud', 'num_users_in_cluster']
    cluster_stats['is_suspicious'] = (cluster_stats['avg_fraud'] > fraud_threshold).astype(int)
    
    # Merge cluster membership with reviews
    df_cluster = df.merge(cluster_membership, on='user_id', how='left').fillna(0)
    
    # Get product-cluster statistics efficiently
    agg_dict = {'user_id': 'count', 'rating': 'mean'} if 'rating' in df.columns else {'user_id': 'count'}
    product_cluster_stats = df_cluster.groupby(['cluster_id', 'product_id']).agg(
        num_reviews_in_cluster=('user_id', 'count'),
        num_cluster_users=('cluster_id', 'nunique'),
        **({'avg_rating_for_product': ('rating', 'mean')} if 'rating' in df.columns else {})
    ).reset_index()
    
    # Add cluster metrics
    product_cluster_stats = product_cluster_stats.merge(
        cluster_stats.reset_index(), on='cluster_id', how='left'
    )
    
    # Get high-fraud users per cluster for comparison
    high_fraud_users = set(cluster_fraud[cluster_fraud['fraud_score'] > fraud_threshold]['user_id'])
    df_cluster['is_fraud_user'] = df_cluster['user_id'].isin(high_fraud_users)
    fraud_product_stats = df_cluster[df_cluster['is_fraud_user']].groupby(
        ['cluster_id', 'product_id']
    ).size().reset_index(name='fraud_reviews')
    
    # Merge fraud stats
    cluster_product_mappings = product_cluster_stats.merge(
        fraud_product_stats, on=['cluster_id', 'product_id'], how='left'
    ).fillna(0)
    
    # Compute derived metrics
    cluster_product_mappings['avg_cluster_reviews_per_user'] = (
        cluster_product_mappings['num_reviews_in_cluster'] / cluster_product_mappings['num_cluster_users']
    ).fillna(0)
    cluster_product_mappings['pct_fraud_reviews'] = (
        cluster_product_mappings['fraud_reviews'] / cluster_product_mappings['num_reviews_in_cluster']
    ).fillna(0)
    
    # Add product rank within cluster
    cluster_product_mappings['rank_in_cluster'] = cluster_product_mappings.groupby('cluster_id')[
        'num_reviews_in_cluster'
    ].rank(method='first', ascending=False).astype(int)
    
    # Rename columns
    cols_to_keep = ['cluster_id', 'product_id', 'num_reviews_in_cluster', 'num_cluster_users',
                    'avg_cluster_reviews_per_user', 'avg_fraud', 'is_suspicious', 'pct_fraud_reviews', 'rank_in_cluster']
    if 'avg_rating_for_product' in cluster_product_mappings.columns:
        cols_to_keep.insert(5, 'avg_rating_for_product')
    cluster_product_mappings = cluster_product_mappings[cols_to_keep]
    
    # Rename for output consistency
    rename_dict = {'avg_fraud': 'avg_cluster_fraud_score', 'is_suspicious': 'is_suspicious_cluster'}
    cluster_product_mappings = cluster_product_mappings.rename(columns=rename_dict)
    
    cluster_product_df = cluster_product_mappings
    
    if len(cluster_product_df) == 0:
        logger.warning("No cluster-product mappings found")
        cluster_product_df = pd.DataFrame(columns=[
            'cluster_id', 'product_id', 'num_reviews_in_cluster', 'num_cluster_users',
            'avg_cluster_reviews_per_user', 'avg_rating_for_product', 'avg_cluster_fraud_score',
            'is_suspicious_cluster', 'pct_fraud_reviews', 'rank_in_cluster'
        ])
    else:
        # Sort by fraud score and review count
        cluster_product_df = cluster_product_df.sort_values(
            ['is_suspicious_cluster', 'avg_cluster_fraud_score', 'num_reviews_in_cluster'],
            ascending=[False, False, False]
        ).reset_index(drop=True)
    
    # Save output
    output_path = Path(output_dir) / 'cluster_target_products.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cluster_product_df.to_csv(output_path, index=False)
    
    logger.info(f"Cluster-product mappings saved to {output_path}")
    logger.info(f"Total mappings: {len(cluster_product_df)}")
    logger.info(f"Suspicious clusters: {cluster_product_df['is_suspicious_cluster'].sum()}")
    
    return cluster_product_df


def get_cluster_summary(
    cluster_membership,
    fraud_scores,
    cluster_product_df,
    output_dir='outputs'
):
    """
    Generate summary statistics for each cluster.
    
    Parameters:
    -----------
    cluster_membership : pd.DataFrame
        Cluster assignments
    fraud_scores : pd.DataFrame
        Fraud scores per user
    cluster_product_df : pd.DataFrame
        Cluster-product mappings
    output_dir : str
        Output directory
        
    Returns:
    --------
    pd.DataFrame
        Cluster-level summary statistics
    """
    
    logger.info("Generating cluster summaries (vectorized)...")
    
    # Vectorized cluster summary
    cluster_analysis = cluster_membership.merge(
        fraud_scores[['user_id', 'fraud_score', 'reviews_per_user']],
        on='user_id', how='left'
    ).fillna(0)
    
    # Aggregate by cluster
    cluster_summary = cluster_analysis.groupby('cluster_id').agg({
        'user_id': 'count',
        'fraud_score': ['mean', 'max'],
        'reviews_per_user': ['mean', 'sum']
    }).droplevel(0, axis=1)
    cluster_summary.columns = ['num_users', 'avg_fraud_score', 'max_fraud_score', 'avg_reviews_per_user', 'num_reviews_in_cluster']
    cluster_summary['suspicious_flag'] = (cluster_summary['avg_fraud_score'] > 0.5).astype(int)
    
    # Get product counts per cluster
    num_products = cluster_product_df.groupby('cluster_id')['product_id'].nunique()
    cluster_summary['num_target_products'] = num_products
    cluster_summary = cluster_summary.fillna(0).reset_index()
    
    summary_df = cluster_summary
    summary_df = summary_df.sort_values('avg_fraud_score', ascending=False)
    
    # Save output
    output_path = Path(output_dir) / 'cluster_summary.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)
    
    logger.info(f"Cluster summaries saved to {output_path}")
    
    return summary_df
