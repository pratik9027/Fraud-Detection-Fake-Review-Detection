"""
Text Similarity Features Module
Computes text-based features using TF-IDF and cosine similarity
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def compute_text_similarity(df, output_dir='outputs', max_features=5000):
    """
    Compute text similarity features for users.
    
    Detects users with similar review text (potential fraud indicator).
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataset with columns: user_id, review_text
    output_dir : str
        Output directory to save metrics
    max_features : int
        Maximum number of TF-IDF features
        
    Returns:
    --------
    pd.DataFrame
        User-level text similarity metrics
    """
    
    logger.info("Computing text similarity features...")
    
    # Ensure review_text column exists and is string
    if 'review_text' not in df.columns:
        logger.warning("review_text column not found. Creating dummy text features.")
        return _create_dummy_text_features(df, output_dir)
    
    df['review_text'] = df['review_text'].fillna('').astype(str)
    
    # Remove empty reviews
    df_text = df[df['review_text'].str.len() > 0].copy()
    
    if len(df_text) == 0:
        logger.warning("No valid review text found. Creating dummy text features.")
        return _create_dummy_text_features(df, output_dir)
    
    logger.info(f"Computing TF-IDF for {len(df_text)} reviews with text...")
    
    # Compute TF-IDF vectors
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(df_text['review_text'])
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    except Exception as e:
        logger.error(f"Error computing TF-IDF: {e}")
        return _create_dummy_text_features(df, output_dir)
    
    # Compute user similarity metrics WITHOUT creating full matrix (memory efficient)
    logger.info("Computing user similarity metrics (memory-efficient, per-user)...")
    
    df_text_indexed = df_text.reset_index(drop=True)
    user_sim_data = []
    
    # Process each user separately to avoid full matrix allocation
    for user_id in df['user_id'].unique():
        user_mask = df_text_indexed['user_id'] == user_id
        user_indices = np.where(user_mask)[0]
        num_reviews = len(user_indices)
        
        if num_reviews <= 1:
            avg_similarity = max_similarity = std_similarity = 0.0
        else:
            # Extract only this user's TF-IDF rows
            user_tfidf = tfidf_matrix[user_indices]
            
            # Compute cosine similarity only for this user's reviews
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                user_sims = cosine_similarity(user_tfidf)
                # Get upper triangle to avoid duplicates
                user_sims = user_sims[np.triu_indices_from(user_sims, k=1)]
                avg_similarity = float(np.mean(user_sims)) if len(user_sims) > 0 else 0.0
                max_similarity = float(np.max(user_sims)) if len(user_sims) > 0 else 0.0
                std_similarity = float(np.std(user_sims)) if len(user_sims) > 0 else 0.0
            except Exception as e:
                logger.debug(f"Error computing similarity for user {user_id}: {e}")
                avg_similarity = max_similarity = std_similarity = 0.0
        
        user_sim_data.append([user_id, avg_similarity, max_similarity, std_similarity, num_reviews])
    
    similarity_df = pd.DataFrame(user_sim_data, columns=[
        'user_id', 'avg_text_similarity', 'max_text_similarity', 'std_text_similarity', 'num_reviews_with_text'
    ])
    similarity_df = similarity_df.fillna(0)
    
    # Save output
    output_path = Path(output_dir) / 'user_similarity_metrics.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    similarity_df.to_csv(output_path, index=False)
    logger.info(f"Text similarity metrics saved to {output_path}")
    
    return similarity_df


def _create_dummy_text_features(df, output_dir='outputs'):
    """Create dummy text features when text data is unavailable"""
    logger.warning("Creating dummy text similarity features...")
    
    dummy_df = pd.DataFrame({
        'user_id': df['user_id'].unique(),
        'avg_text_similarity': 0.0,
        'max_text_similarity': 0.0,
        'std_text_similarity': 0.0,
        'num_reviews_with_text': 0
    })
    
    output_path = Path(output_dir) / 'user_similarity_metrics.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_df.to_csv(output_path, index=False)
    
    return dummy_df
