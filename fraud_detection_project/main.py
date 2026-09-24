"""
Fraud Detection Pipeline - Main Orchestrator
Coordinates all modules to execute the complete fraud detection workflow
"""

import logging
import sys
from pathlib import Path
import pandas as pd

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.data_loader import load_dataset
from backend.behavioral_features import compute_behavior_features
from backend.text_features import compute_text_similarity
from backend.rating_features import compute_rating_entropy
from backend.graph_analysis import build_user_graph, compute_graph_metrics
from backend.fraud_scoring import compute_fraud_score
from backend.cluster_analysis import map_cluster_to_products, get_cluster_summary

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_pipeline(
    data_source=None,
    output_dir='outputs',
    fraud_threshold=0.5,
    weights=None
):
    """
    Execute the complete fraud detection pipeline.
    
    Parameters:
    -----------
    data_source : str or None
        Path to dataset (JSON or CSV). If None, uses sample dataset.
    output_dir : str
        Directory to save outputs
    fraud_threshold : float
        Fraud score threshold for flagging suspicious users
    weights : dict
        Custom weights for fraud score components
        
    Returns:
    --------
    dict : Pipeline results containing all computed metrics and scores
    """
    
    logger.info("=" * 80)
    logger.info("STARTING FRAUD DETECTION PIPELINE")
    logger.info("=" * 80)
    
    results = {}
    
    try:
        # STEP 1: Load dataset
        logger.info("\n[STEP 1] Loading and preprocessing dataset...")
        df = load_dataset(
            json_file=data_source if data_source and data_source.endswith('.json') else None,
            csv_file=data_source if data_source and data_source.endswith('.csv') else None,
            output_dir=output_dir
        )
        results['dataset'] = df
        logger.info(f"✓ Dataset loaded: {df.shape[0]} reviews from {df['user_id'].nunique()} users")
        
        # STEP 2: Compute behavioral features
        logger.info("\n[STEP 2] Computing behavioral features...")
        behavior_metrics = compute_behavior_features(df, output_dir)
        results['behavior_metrics'] = behavior_metrics
        logger.info(f"✓ Behavioral features computed for {len(behavior_metrics)} users")
        
        # STEP 3: Compute text similarity features
        logger.info("\n[STEP 3] Computing text similarity features...")
        text_similarity = compute_text_similarity(df, output_dir)
        results['text_similarity'] = text_similarity
        logger.info(f"✓ Text similarity computed for {len(text_similarity)} users")
        
        # STEP 4: Compute rating entropy features
        logger.info("\n[STEP 4] Computing rating entropy features...")
        rating_metrics = compute_rating_entropy(df, output_dir)
        results['rating_metrics'] = rating_metrics
        logger.info(f"✓ Rating entropy computed for {len(rating_metrics)} users")
        
        # STEP 5: Build user interaction graph
        logger.info("\n[STEP 5] Building user interaction graph...")
        graph_data = build_user_graph(df, output_dir, min_common_products=1)
        results['graph_data'] = graph_data
        logger.info(f"✓ Graph built with {len(graph_data['graph'].nodes())} nodes and {len(graph_data['graph'].edges())} edges")
        logger.info(f"✓ Found {len(graph_data['communities'])} clusters")
        
        # STEP 6: Compute graph-based metrics
        logger.info("\n[STEP 6] Computing graph-based metrics...")
        graph_metrics = compute_graph_metrics(df, graph_data, output_dir)
        results['graph_metrics'] = graph_metrics
        logger.info(f"✓ Graph metrics computed for {len(graph_metrics)} users")
        
        # STEP 7: Compute fraud scores
        logger.info("\n[STEP 7] Computing composite fraud scores...")
        fraud_scores = compute_fraud_score(
            behavior_metrics,
            text_similarity,
            rating_metrics,
            graph_metrics,
            output_dir,
            weights=weights
        )
        results['fraud_scores'] = fraud_scores
        
        # Fraud statistics
        num_suspicious = (fraud_scores['fraud_score'] > fraud_threshold).sum()
        logger.info(f"✓ Fraud scores computed")
        logger.info(f"  - Suspicious users (score > {fraud_threshold}): {num_suspicious}")
        logger.info(f"  - Fraud score range: [{fraud_scores['fraud_score'].min():.4f}, {fraud_scores['fraud_score'].max():.4f}]")
        
        # STEP 8: Map clusters to products
        logger.info("\n[STEP 8] Mapping clusters to targeted products...")
        cluster_products = map_cluster_to_products(
            df,
            graph_data['cluster_membership'],
            fraud_scores,
            output_dir,
            fraud_threshold
        )
        results['cluster_products'] = cluster_products
        logger.info(f"✓ {len(cluster_products)} cluster-product mappings created")
        
        # STEP 9: Generate cluster summaries
        logger.info("\n[STEP 9] Generating cluster summaries...")
        cluster_summary = get_cluster_summary(
            graph_data['cluster_membership'],
            fraud_scores,
            cluster_products,
            output_dir
        )
        results['cluster_summary'] = cluster_summary
        logger.info(f"✓ Cluster summary generated")
        
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        logger.info(f"\nOutput files saved to: {Path(output_dir).absolute()}")
        logger.info("\nGenerated files:")
        logger.info("  • clean_dataset.csv")
        logger.info("  • user_behavior_metrics.csv")
        logger.info("  • user_similarity_metrics.csv")
        logger.info("  • user_rating_metrics.csv")
        logger.info("  • cluster_membership.csv")
        logger.info("  • cluster_info.csv")
        logger.info("  • user_graph_metrics.csv")
        logger.info("  • user_fraud_scores.csv")
        logger.info("  • cluster_target_products.csv")
        logger.info("  • cluster_summary.csv")
        
        return results
        
    except Exception as e:
        logger.error(f"\n✗ Pipeline failed with error: {e}", exc_info=True)
        raise


def print_summary(results):
    """Print summary statistics of pipeline results"""
    
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 80)
    
    if 'dataset' in results:
        df = results['dataset']
        logger.info(f"\nDataset Statistics:")
        logger.info(f"  Total Reviews: {len(df)}")
        logger.info(f"  Unique Users: {df['user_id'].nunique()}")
        logger.info(f"  Unique Products: {df['product_id'].nunique()}")
        logger.info(f"  Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    if 'fraud_scores' in results:
        fraud_scores = results['fraud_scores']
        logger.info(f"\nFraud Score Summary:")
        logger.info(f"  Mean Score: {fraud_scores['fraud_score'].mean():.4f}")
        logger.info(f"  Std Dev: {fraud_scores['fraud_score'].std():.4f}")
        logger.info(f"  Min Score: {fraud_scores['fraud_score'].min():.4f}")
        logger.info(f"  Max Score: {fraud_scores['fraud_score'].max():.4f}")
        logger.info(f"  High Risk Users (> 0.7): {(fraud_scores['fraud_score'] > 0.7).sum()}")
        logger.info(f"  Medium Risk Users (0.5-0.7): {((fraud_scores['fraud_score'] >= 0.5) & (fraud_scores['fraud_score'] <= 0.7)).sum()}")
    
    if 'cluster_summary' in results:
        clusters = results['cluster_summary']
        logger.info(f"\nCluster Summary:")
        logger.info(f"  Total Clusters: {len(clusters)}")
        logger.info(f"  Suspicious Clusters: {clusters['suspicious_flag'].sum()}")
        logger.info(f"  Avg Cluster Size: {clusters['num_users'].mean():.1f} users")
    
    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    """
    Main execution point
    
    Usage:
        python main.py [data_file] [output_dir]
    
    Examples:
        python main.py                                    # Use sample dataset
        python main.py reviews_Electronics_5.json         # Use JSON file
        python main.py ratings_Electronics.csv outputs/   # Use CSV file with custom output dir
    """
    
    # Parse command line arguments
    data_file = sys.argv[1] if len(sys.argv) > 1 else None
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'outputs'
    
    # Run pipeline
    results = run_pipeline(
        data_source=data_file,
        output_dir=output_dir
    )
    
    # Print summary
    print_summary(results)
