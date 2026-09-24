"""
Fraud Detection Dashboard - Streamlit Frontend
Interactive visualization and exploration of fraud detection results
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import logging

# Configure page
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_data
def load_data(output_dir='outputs'):
    """Load all required CSV files"""
    try:
        fraud_scores = pd.read_csv(Path(output_dir) / 'user_fraud_scores.csv')
        cluster_membership = pd.read_csv(Path(output_dir) / 'cluster_membership.csv')
        cluster_products = pd.read_csv(Path(output_dir) / 'cluster_target_products.csv')
        cluster_summary = pd.read_csv(Path(output_dir) / 'cluster_summary.csv')
        clean_dataset = pd.read_csv(Path(output_dir) / 'clean_dataset.csv')
        
        return {
            'fraud_scores': fraud_scores,
            'cluster_membership': cluster_membership,
            'cluster_products': cluster_products,
            'cluster_summary': cluster_summary,
            'clean_dataset': clean_dataset
        }
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        st.error(f"Error loading data files. Make sure you've run the pipeline first.")
        return None


def display_header():
    """Display dashboard header"""
    st.title("🔍 Fraud Detection Dashboard")
    st.markdown("""
    Coordinated Fake Review Detection System for Amazon Electronics Reviews
    
    This dashboard analyzes review patterns to identify suspicious reviewers and coordinated fraud rings.
    """)


def display_overview_metrics(data):
    """Display key overview metrics"""
    fraud_scores = data['fraud_scores']
    cluster_summary = data['cluster_summary']
    clean_dataset = data['clean_dataset']
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Reviews",
            f"{len(clean_dataset):,}",
            delta="Dataset size"
        )
    
    with col2:
        st.metric(
            "Unique Users",
            f"{len(fraud_scores):,}",
            delta="Analyzed users"
        )
    
    with col3:
        high_risk = (fraud_scores['fraud_score'] > 0.7).sum()
        st.metric(
            "High Risk Users",
            f"{high_risk:,}",
            delta=f"{(high_risk/len(fraud_scores)*100):.1f}%"
        )
    
    with col4:
        medium_risk = ((fraud_scores['fraud_score'] >= 0.5) & (fraud_scores['fraud_score'] <= 0.7)).sum()
        st.metric(
            "Medium Risk Users",
            f"{medium_risk:,}",
            delta=f"{(medium_risk/len(fraud_scores)*100):.1f}%"
        )
    
    with col5:
        st.metric(
            "Clusters Detected",
            f"{len(cluster_summary):,}",
            delta=f"{cluster_summary['suspicious_flag'].sum()} suspicious"
        )


def display_fraud_score_distribution(data):
    """Display fraud score histogram with Plotly"""
    fraud_scores = data['fraud_scores']
    
    st.subheader("📊 Fraud Score Distribution")
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=fraud_scores['fraud_score'],
        nbinsx=50,
        name='Fraud Score',
        marker_color='indianred',
        hovertemplate='<b>Score Range</b>: %{x}<br><b>Count</b>: %{y}<extra></extra>'
    ))
    
    # Add threshold lines
    fig.add_vline(
        x=0.7,
        line_dash="dash",
        line_color="red",
        annotation_text="High Risk (0.7)",
        annotation_position="top right"
    )
    fig.add_vline(
        x=0.5,
        line_dash="dash",
        line_color="orange",
        annotation_text="Medium Risk (0.5)",
        annotation_position="top left"
    )
    
    fig.update_layout(
        title="Distribution of User Fraud Scores",
        xaxis_title="Fraud Score",
        yaxis_title="Number of Users",
        hovermode='x unified',
        height=400,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)


def display_top_suspicious_users(data):
    """Display top suspicious users table"""
    fraud_scores = data['fraud_scores']
    
    st.subheader("🚨 Top Suspicious Users")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        num_users = st.slider("Number of users to display:", min_value=5, max_value=100, value=20)
    
    with col2:
        show_all_metrics = st.checkbox("Show all metrics", value=False)
    
    top_users = fraud_scores.head(num_users).copy()
    
    if show_all_metrics:
        display_df = top_users
    else:
        display_df = top_users[['rank', 'user_id', 'fraud_score', 'behavior_score', 
                                 'similarity_score', 'rating_entropy_score', 'cluster_score', 'reviews_per_user']]
    
    # Format numeric columns
    display_df = display_df.copy()
    for col in display_df.select_dtypes(include='float').columns:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    st.dataframe(display_df, use_container_width=True, height=400)
    
    # Download button
    csv = top_users.to_csv(index=False)
    st.download_button(
        label="📥 Download Top Users (CSV)",
        data=csv,
        file_name=f"top_{num_users}_suspicious_users.csv",
        mime="text/csv"
    )


def display_score_components_analysis(data):
    """Display analysis of fraud score components"""
    fraud_scores = data['fraud_scores']
    
    st.subheader("📈 Fraud Score Components Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Component contributions
        components = {
            'Behavior Score': fraud_scores['behavior_score'].mean(),
            'Similarity Score': fraud_scores['similarity_score'].mean(),
            'Rating Entropy Score': fraud_scores['rating_entropy_score'].mean(),
            'Cluster Score': fraud_scores['cluster_score'].mean()
        }
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=list(components.keys()),
            values=list(components.values()),
            marker_colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        )])
        
        fig_pie.update_layout(
            title="Average Contribution of Score Components",
            height=400
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Component distributions
        fig_box = go.Figure()
        
        for component in ['behavior_score', 'similarity_score', 'rating_entropy_score', 'cluster_score']:
            fig_box.add_trace(go.Box(
                y=fraud_scores[component],
                name=component.replace('_', ' ').title()
            ))
        
        fig_box.update_layout(
            title="Distribution of Score Components",
            yaxis_title="Score Value",
            height=400,
            showlegend=True
        )
        
        st.plotly_chart(fig_box, use_container_width=True)


def display_cluster_analysis(data):
    """Display cluster-level analysis"""
    cluster_summary = data['cluster_summary']
    cluster_products = data['cluster_products']
    
    st.subheader("🔗 Cluster Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Cluster Summary Statistics**")
        
        summary_stats = {
            'Total Clusters': len(cluster_summary),
            'Suspicious Clusters': int(cluster_summary['suspicious_flag'].sum()),
            'Avg Users per Cluster': f"{cluster_summary['num_users'].mean():.1f}",
            'Max Users in Cluster': int(cluster_summary['num_users'].max()),
            'Avg Fraud Score': f"{cluster_summary['avg_fraud_score'].mean():.4f}"
        }
        
        for metric, value in summary_stats.items():
            st.metric(metric, value)
    
    with col2:
        st.write("**Cluster Risk Distribution**")
        
        fig_cluster = px.bar(
            cluster_summary,
            x='cluster_id',
            y='avg_fraud_score',
            color='suspicious_flag',
            color_discrete_map={0: '#90EE90', 1: '#FF6B6B'},
            labels={'cluster_id': 'Cluster ID', 'avg_fraud_score': 'Avg Fraud Score'},
            height=300
        )
        
        st.plotly_chart(fig_cluster, use_container_width=True)
    
    # Cluster details table
    st.write("**Detailed Cluster Information**")
    
    cluster_display = cluster_summary.copy()
    cluster_display['Risk'] = cluster_display['suspicious_flag'].apply(
        lambda x: '🔴 High' if x == 1 else '🟢 Low'
    )
    
    st.dataframe(
        cluster_display[['cluster_id', 'num_users', 'num_target_products', 'avg_fraud_score', 'Risk']],
        use_container_width=True,
        height=300
    )


def display_targeted_products(data):
    """Display products targeted by fraud rings"""
    cluster_products = data['cluster_products']
    
    st.subheader("🎯 Targeted Products Analysis")
    
    # Filter suspicious products
    suspicious_products = cluster_products[cluster_products['is_suspicious_cluster'] == 1]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write(f"**Suspicious Product-Cluster Relationships: {len(suspicious_products)}**")
    
    with col2:
        sort_by = st.selectbox(
            "Sort by:",
            ['num_reviews_in_cluster', 'avg_cluster_fraud_score', 'avg_rating_for_product'],
            index=0
        )
    
    if len(suspicious_products) > 0:
        suspicious_products_sorted = suspicious_products.sort_values(sort_by, ascending=False).head(50)
        
        display_products = suspicious_products_sorted[[
            'cluster_id', 'product_id', 'num_reviews_in_cluster',
            'num_cluster_users', 'avg_cluster_fraud_score', 'avg_rating_for_product'
        ]].copy()
        
        display_products['avg_cluster_fraud_score'] = display_products['avg_cluster_fraud_score'].apply(lambda x: f"{x:.4f}")
        display_products['avg_rating_for_product'] = display_products['avg_rating_for_product'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(display_products, use_container_width=True, height=400)
        
        # Download button
        csv = suspicious_products_sorted.to_csv(index=False)
        st.download_button(
            label="📥 Download Suspicious Products (CSV)",
            data=csv,
            file_name="suspicious_targeted_products.csv",
            mime="text/csv"
        )
    else:
        st.info("No suspicious targeted products found")


def display_user_search(data):
    """Interactive user search and details"""
    fraud_scores = data['fraud_scores']
    cluster_membership = data['cluster_membership']
    clean_dataset = data['clean_dataset']
    
    st.subheader("🔎 Search User Details")
    
    user_id = st.selectbox(
        "Select a user:",
        sorted(fraud_scores['user_id'].unique()),
        format_func=lambda x: f"{x} (Fraud Score: {fraud_scores[fraud_scores['user_id']==x]['fraud_score'].values[0]:.4f})"
    )
    
    if user_id:
        user_fraud = fraud_scores[fraud_scores['user_id'] == user_id].iloc[0]
        user_cluster = cluster_membership[cluster_membership['user_id'] == user_id].iloc[0] if len(cluster_membership[cluster_membership['user_id'] == user_id]) > 0 else None
        user_reviews = clean_dataset[clean_dataset['user_id'] == user_id]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Fraud Score", f"{user_fraud['fraud_score']:.4f}")
            st.metric("Reviews Count", int(user_fraud['reviews_per_user']))
        
        with col2:
            st.metric("Behavior Score", f"{user_fraud['behavior_score']:.4f}")
            st.metric("Similarity Score", f"{user_fraud['similarity_score']:.4f}")
        
        with col3:
            st.metric("Rating Entropy Score", f"{user_fraud['rating_entropy_score']:.4f}")
            st.metric("Cluster Score", f"{user_fraud['cluster_score']:.4f}")
        
        if user_cluster is not None:
            st.write(f"**Cluster ID:** {int(user_cluster['cluster_id'])}")
        
        st.write(f"**Review Timeline:** {len(user_reviews)} reviews")
        
        # Show sample reviews
        if len(user_reviews) > 0:
            st.write("**Sample Reviews:**")
            sample_cols = ['product_id', 'rating']
            if 'review_text' in user_reviews.columns:
                sample_cols.append('review_text')
            if 'timestamp' in user_reviews.columns:
                sample_cols.append('timestamp')
            
            sample_reviews = user_reviews[sample_cols].head(10)
            st.dataframe(sample_reviews, use_container_width=True)


def display_statistics_export(data):
    """Display statistics and export options"""
    st.subheader("📊 Data Export")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fraud_csv = data['fraud_scores'].to_csv(index=False)
        st.download_button(
            label="📥 All Fraud Scores",
            data=fraud_csv,
            file_name="user_fraud_scores.csv",
            mime="text/csv"
        )
    
    with col2:
        cluster_csv = data['cluster_products'].to_csv(index=False)
        st.download_button(
            label="📥 Cluster Products",
            data=cluster_csv,
            file_name="cluster_target_products.csv",
            mime="text/csv"
        )
    
    with col3:
        summary_csv = data['cluster_summary'].to_csv(index=False)
        st.download_button(
            label="📥 Cluster Summary",
            data=summary_csv,
            file_name="cluster_summary.csv",
            mime="text/csv"
        )


def main():
    """Main app entry point"""
    
    # Sidebar
    st.sidebar.title("⚙️ Configuration")
    output_dir = st.sidebar.text_input("Output Directory:", value="outputs")
    
    # Load data
    with st.spinner("Loading data..."):
        data = load_data(output_dir)
    
    if data is None:
        st.stop()
    
    # Display sections
    display_header()
    display_overview_metrics(data)
    
    st.divider()
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📈 Analysis", "🚨 Suspicious Users", "🔗 Clusters", "🎯 Products"]
    )
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            display_fraud_score_distribution(data)
        with col2:
            display_score_components_analysis(data)
    
    with tab2:
        display_top_suspicious_users(data)
        st.divider()
        display_user_search(data)
    
    with tab3:
        display_cluster_analysis(data)
    
    with tab4:
        display_targeted_products(data)
    
    st.divider()
    
    # Export section
    with st.expander("📦 Export Data"):
        display_statistics_export(data)
    
    # Footer
    st.markdown("""
    ---
    **Fraud Detection System** | Amazon Electronics Reviews Analysis
    
    Generated outputs:
    - User behavioral metrics
    - Text similarity analysis
    - Rating entropy calculations
    - Network clustering
    - Fraud risk scores
    """)


if __name__ == "__main__":
    main()
