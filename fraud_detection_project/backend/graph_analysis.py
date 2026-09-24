"""
Optimized Graph Analysis Module
Builds user interaction graphs and computes fast graph-based metrics
"""

import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
import logging
from itertools import combinations
from collections import defaultdict

logger = logging.getLogger(__name__)


# ===============================
# GRAPH BUILDING
# ===============================

def build_user_graph(df, output_dir="outputs", min_common_products=3):
    """
    Build user interaction graph efficiently using inverted index approach.

    Nodes: users
    Edges: users connected if they reviewed >= min_common_products same products
    """

    logger.info("Building optimized user interaction graph...")

    G = nx.Graph()

    users = df["user_id"].unique()
    G.add_nodes_from(users)

    logger.info(f"Added {len(users)} user nodes")

    # Build product -> users mapping
    product_users = df.groupby("product_id")["user_id"].apply(list)

    edge_weights = defaultdict(int)

    logger.info("Creating edges using product-user inverted index...")

    # Efficient edge generation
    for users_list in product_users:

        if len(users_list) < 2:
            continue

        for u1, u2 in combinations(users_list, 2):
            edge = tuple(sorted((u1, u2)))
            edge_weights[edge] += 1

    # Add edges meeting threshold
    edge_count = 0

    for (u1, u2), weight in edge_weights.items():

        if weight >= min_common_products:
            G.add_edge(u1, u2, weight=weight)
            edge_count += 1

    logger.info(f"Added {edge_count} edges")
    logger.info(f"Graph density: {nx.density(G):.5f}")

    # Community detection
    logger.info("Detecting communities (greedy modularity)...")

    try:
        communities = list(nx.community.greedy_modularity_communities(G))

    except Exception as e:
        logger.warning(f"Community detection failed: {e}")
        communities = [set(G.nodes())]

    logger.info(f"Detected {len(communities)} communities")

    # Cluster membership mapping
    cluster_mapping = {}

    for cluster_id, community in enumerate(communities):
        for user in community:
            cluster_mapping[user] = cluster_id

    cluster_membership = pd.DataFrame({
        "user_id": list(cluster_mapping.keys()),
        "cluster_id": list(cluster_mapping.values())
    })

    # Cluster statistics
    cluster_info = []

    for cluster_id, community in enumerate(communities):

        cluster_users = list(community)
        cluster_df = df[df["user_id"].isin(cluster_users)]

        cluster_info.append({
            "cluster_id": cluster_id,
            "num_users": len(cluster_users),
            "num_reviews": len(cluster_df),
            "num_products": cluster_df["product_id"].nunique(),
            "avg_reviews_per_user":
                len(cluster_df) / len(cluster_users)
                if len(cluster_users) > 0 else 0,
            "clustering_coefficient":
                nx.average_clustering(G.subgraph(cluster_users))
                if len(cluster_users) > 1 else 0
        })

    cluster_info_df = pd.DataFrame(cluster_info)

    # Save outputs
    output_path_membership = Path(output_dir) / "cluster_membership.csv"
    output_path_info = Path(output_dir) / "cluster_info.csv"

    output_path_membership.parent.mkdir(parents=True, exist_ok=True)

    cluster_membership.to_csv(output_path_membership, index=False)
    cluster_info_df.to_csv(output_path_info, index=False)

    logger.info("Cluster membership saved")
    logger.info("Cluster info saved")

    return {
        "graph": G,
        "cluster_membership": cluster_membership,
        "cluster_info": cluster_info_df,
        "communities": communities
    }


# ===============================
# GRAPH METRICS
# ===============================

def compute_graph_metrics(df, graph_data, output_dir="outputs"):
    """
    Compute user-level graph metrics efficiently
    """

    logger.info("Computing optimized graph metrics...")

    G = graph_data["graph"]
    cluster_membership = graph_data["cluster_membership"]

    # Precompute expensive centralities ONCE
    logger.info("Computing degree centrality...")
    degree_centrality_dict = nx.degree_centrality(G)

    logger.info("Computing approximate betweenness centrality...")
    betweenness_dict = nx.betweenness_centrality(G, k=min(50, len(G)))

    logger.info("Computing clustering coefficients...")
    clustering_dict = nx.clustering(G)

    graph_metrics = []

    cluster_lookup = dict(
        zip(cluster_membership.user_id,
            cluster_membership.cluster_id)
    )

    for user_id in G.nodes():

        graph_metrics.append({

            "user_id": user_id,

            "cluster_id":
                cluster_lookup.get(user_id, -1),

            "degree":
                G.degree(user_id),

            "degree_centrality":
                degree_centrality_dict.get(user_id, 0),

            "betweenness_centrality":
                betweenness_dict.get(user_id, 0),

            "clustering_coefficient":
                clustering_dict.get(user_id, 0)
        })

    graph_metrics_df = pd.DataFrame(graph_metrics)

    output_path = Path(output_dir) / "user_graph_metrics.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    graph_metrics_df.to_csv(output_path, index=False)

    logger.info("Graph metrics saved successfully")

    return graph_metrics_df