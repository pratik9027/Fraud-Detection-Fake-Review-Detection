"""
Data Loading and Preprocessing Module
Handles loading, cleaning, and formatting Amazon reviews dataset
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(json_file=None, csv_file=None, output_dir='outputs'):
    """
    Load and preprocess Amazon reviews dataset.
    
    Parameters:
    -----------
    json_file : str
        Path to JSON dataset file (reviews_Electronics_5.json)
    csv_file : str
        Path to CSV dataset file (ratings_Electronics.csv)
    output_dir : str
        Output directory to save cleaned dataset
        
    Returns:
    --------
    pd.DataFrame
        Cleaned dataset with standardized schema
    """
    
    logger.info("Loading dataset...")
    
    # Load from JSON if provided
    if json_file and Path(json_file).exists():
        df = _load_json_dataset(json_file)
    elif csv_file and Path(csv_file).exists():
        df = _load_csv_dataset(csv_file)
    else:
        logger.warning("No valid dataset file provided. Creating sample dataset for demonstration.")
        df = _create_sample_dataset()
    
    # Standardize column names
    column_mapping = {
        'reviewerID': 'user_id',
        'asin': 'product_id',
        'reviewText': 'review_text',
        'overall': 'rating',
        'unixReviewTime': 'timestamp'
    }
    
    df = df.rename(columns=column_mapping)
    
    # Select relevant columns
    required_cols = ['user_id', 'product_id', 'review_text', 'rating', 'timestamp']
    available_cols = [col for col in required_cols if col in df.columns]
    df = df[available_cols]
    
    # Convert timestamp from unix time to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Remove missing values
    logger.info(f"Removing missing values. Original shape: {df.shape}")
    df = df.dropna(subset=['user_id', 'product_id', 'rating'])
    logger.info(f"Shape after removing NaN: {df.shape}")
    
    # Handle missing review text
    if 'review_text' in df.columns:
        df['review_text'] = df['review_text'].fillna('')
    
    # Sort chronologically
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp').reset_index(drop=True)
        logger.info("Dataset sorted chronologically")
    
    # Convert rating to numeric
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df = df.dropna(subset=['rating'])
    
    # Save cleaned dataset
    output_path = Path(output_dir) / 'clean_dataset.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Cleaned dataset saved to {output_path}")
    logger.info(f"Final dataset shape: {df.shape}")
    
    return df


def _load_json_dataset(json_file):
    """Load dataset from JSON Lines format (optimized with chunking)"""
    logger.info(f"Loading JSON dataset from {json_file}...")
    records = []
    chunk_size = 5000
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                records.append(json.loads(line))
                if len(records) % chunk_size == 0:
                    logger.info(f"Loaded {len(records)} records...")
            except json.JSONDecodeError:
                continue
    
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} records from JSON")
    return df


def _load_csv_dataset(csv_file):
    """Load dataset from CSV format"""
    logger.info(f"Loading CSV dataset from {csv_file}...")
    df = pd.read_csv(csv_file)
    logger.info(f"Loaded {len(df)} records from CSV")
    return df


def _create_sample_dataset(num_records=10000):
    """
    Create sample dataset for demonstration and testing.
    Includes synthetic fraud patterns.
    """
    logger.info("Creating sample dataset for demonstration...")
    
    np.random.seed(42)
    
    # Create base users (mostly legitimate)
    num_legit_users = 800
    num_fraud_users = 50
    num_products = 500
    
    records = []
    
    # Legitimate users - normal review patterns
    for user_id in range(1, num_legit_users + 1):
        num_reviews = np.random.randint(1, 20)
        for _ in range(num_reviews):
            records.append({
                'reviewerID': f'user_{user_id}',
                'asin': f'prod_{np.random.randint(1, num_products)}',
                'reviewText': f'Good product. {np.random.choice(["Love it", "Works well", "Great quality", "Recommended"])}',
                'overall': np.random.choice([3, 4, 5], p=[0.2, 0.3, 0.5]),
                'unixReviewTime': int(datetime(2023, 1, 1).timestamp()) + np.random.randint(0, 86400 * 365)
            })
    
    # Fraud users - coordinated review patterns
    fraud_products = [f'prod_{np.random.randint(1, 50)}' for _ in range(10)]  # Targeted products
    
    for user_id in range(num_legit_users + 1, num_legit_users + num_fraud_users + 1):
        num_reviews = np.random.randint(50, 150)  # Many reviews
        for _ in range(num_reviews):
            records.append({
                'reviewerID': f'user_{user_id}',
                'asin': np.random.choice(fraud_products),  # Same products
                'reviewText': 'Great product!!!',  # Repetitive text
                'overall': 5,  # Always 5 stars
                'unixReviewTime': int(datetime(2023, 1, 1).timestamp()) + np.random.randint(0, 86400 * 365)
            })
    
    df = pd.DataFrame(records)
    logger.info(f"Created sample dataset with {len(df)} reviews from {len(df['reviewerID'].unique())} users")
    
    return df
