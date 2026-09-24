# Fraud Detection System - Amazon Electronics Reviews

A comprehensive, modular Python system for detecting coordinated fake reviewers and fraud rings in Amazon product reviews using behavioral analysis, text similarity, rating patterns, and graph-based clustering.

## 📋 Project Structure

```
fraud_detection_project/
├── data/
│   └── reviews.csv                    # Raw dataset
│
├── backend/
│   ├── __init__.py
│   ├── data_loader.py                 # Data loading & preprocessing
│   ├── behavioral_features.py         # User activity metrics
│   ├── text_features.py               # TF-IDF & similarity analysis
│   ├── rating_features.py             # Rating entropy calculation
│   ├── graph_analysis.py              # Network graph & clustering
│   ├── fraud_scoring.py               # Composite fraud scoring
│   └── cluster_analysis.py            # Cluster-to-product mapping
│
├── outputs/                           # Generated analysis files
│   ├── clean_dataset.csv
│   ├── user_behavior_metrics.csv
│   ├── user_similarity_metrics.csv
│   ├── user_rating_metrics.csv
│   ├── cluster_membership.csv
│   ├── cluster_info.csv
│   ├── user_graph_metrics.csv
│   ├── user_fraud_scores.csv
│   ├── cluster_target_products.csv
│   └── cluster_summary.csv
│
├── frontend/
│   └── app.py                         # Streamlit dashboard
│
├── main.py                            # Pipeline orchestrator
├── data.py                            # Dataset download script
└── requirements.txt                   # Dependencies
```

## 🔧 Installation

### 1. Clone/Create Project
```bash
cd fraud_detection_project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Data (Optional)
```bash
python data.py
```

This will download:
- Amazon Electronics reviews (JSON format)
- Product metadata (JSON format)
- Rating CSV

## ▶️ Running the Pipeline

### Basic Usage (Sample Data)
```bash
python main.py
```

This will create a sample dataset with synthetic fraud patterns for testing.

### With Your Own Data
```bash
# Using JSON file
python main.py path/to/reviews_Electronics_5.json

# Using CSV file
python main.py path/to/ratings_Electronics.csv outputs/
```

## 📊 Pipeline Stages

### Step 1: Data Loading
- Reads JSON or CSV dataset
- Renames columns to standard schema:
  - `reviewerID` → `user_id`
  - `asin` → `product_id`
  - `reviewText` → `review_text`
  - `overall` → `rating`
  - `unixReviewTime` → `timestamp`
- Converts unix timestamps to datetime
- Removes missing values
- Sorts chronologically
- **Output:** `outputs/clean_dataset.csv`

### Step 2: Behavioral Features
Detects suspicious review patterns:
- `reviews_per_user`: Total review count (higher = more active)
- `reviews_per_day`: Activity rate (higher = concentrated activity)
- `burst_activity_score`: Concentration in time windows (higher = burst behavior)
- `unique_products`: Diversity of reviewed products (lower = targeted behavior)
- `product_concentration`: Ratio of repeated products (higher = suspicious)
- `avg_hours_between_reviews`: Time between reviews (lower = unusual pattern)
- **Output:** `outputs/user_behavior_metrics.csv`

### Step 3: Text Similarity Features
Identifies reviewers with similar review text:
- Uses TF-IDF vectorization on review text
- Computes cosine similarity between reviews
- For each user:
  - `avg_text_similarity`: Average similarity to own reviews
  - `max_text_similarity`: Maximum similarity
  - `std_text_similarity`: Variation in text
- **Output:** `outputs/user_similarity_metrics.csv`

### Step 4: Rating Entropy
Analyzes rating distribution to find unnatural patterns:
- Shannon entropy of rating distribution (normalized)
- `rating_entropy`: High = diverse ratings (legitimate), Low = concentrated (suspicious)
- `rating_concentration`: 1 - entropy (inverse measure)
- `pct_five_star`: Percentage of 5-star reviews (fraud indicator)
- `pct_positive_ratings`: Percentage of 4-5 star reviews
- **Output:** `outputs/user_rating_metrics.csv`

### Step 5: Graph Building & Clustering
Creates network of user interactions:
- Nodes: Individual users
- Edges: Connect users if they reviewed ≥1 same products
- Weighted edges by number of common products
- Community detection using greedy modularity algorithm
- Identifies review fraud rings (tight communities)
- **Output:** 
  - `outputs/cluster_membership.csv`
  - `outputs/cluster_info.csv`

### Step 6: Graph Metrics
Computes user-level network metrics:
- `degree`: Number of connections
- `degree_centrality`: Normalized connectivity
- `betweenness_centrality`: Bridge role in network
- `clustering_coefficient`: Cohesion with neighbors
- **Output:** `outputs/user_graph_metrics.csv`

### Step 7: Fraud Scoring
Combines all features into composite fraud score:

**Formula:**
```
fraud_score = 0.3 * behavior_score
            + 0.25 * similarity_score
            + 0.2 * rating_entropy_score
            + 0.25 * cluster_score
```

All component scores are normalized to [0, 1] using MinMaxScaler.

Score Interpretation:
- **0.0 - 0.3**: Low risk (legitimate reviewer)
- **0.3 - 0.5**: Medium risk (some suspicious patterns)
- **0.5 - 0.7**: High risk (multiple fraud indicators)
- **0.7 - 1.0**: Very high risk (strong fraud signals)

**Output:** `outputs/user_fraud_scores.csv`

### Step 8: Cluster-to-Product Mapping
Maps which products are targeted by each fraud cluster:
- For each cluster, identifies targeted products
- Calculates review concentration per product
- Flags suspicious product-cluster pairs
- **Output:** `outputs/cluster_target_products.csv`

### Step 9: Cluster Summary
Generates cluster-level statistics:
- Cluster size and composition
- Average fraud score per cluster
- Number of target products
- Suspicious flag
- **Output:** `outputs/cluster_summary.csv`

## 🎨 Streamlit Dashboard

Launch the interactive dashboard:
```bash
streamlit run frontend/app.py
```

### Dashboard Features

#### Overview Tab
- Key metrics: Total reviews, users, suspicious users
- Fraud score distribution histogram
- Score component contributions (pie chart)
- Distribution of component scores (box plots)

#### Suspicious Users Tab
- Top N suspicious users table (configurable)
- All fraud metrics displayed
- Individual user detail view
- Search and filter by user ID
- Download user data

#### Cluster Analysis Tab
- Cluster risk distribution
- Cluster statistics and details
- Suspicious vs. legitimate cluster comparison
- Cluster metrics visualization

#### Products Tab
- Products targeted by fraud rings
- Review concentration analysis
- Product-cluster relationships
- Sort by fraud score, review count, or rating

#### Data Export
- Download full fraud scores
- Export cluster products
- Export cluster summary
- CSV format for further analysis

## 📈 Output Files Description

| File | Contents | Key Columns |
|------|----------|-------------|
| `clean_dataset.csv` | Preprocessed reviews | user_id, product_id, rating, timestamp |
| `user_behavior_metrics.csv` | Activity patterns | reviews_per_user, burst_activity_score, product_concentration |
| `user_similarity_metrics.csv` | Text analysis | avg_text_similarity, max_text_similarity |
| `user_rating_metrics.csv` | Rating patterns | rating_entropy, pct_five_star |
| `cluster_membership.csv` | Community detection | user_id, cluster_id |
| `user_graph_metrics.csv` | Network metrics | degree_centrality, clustering_coefficient |
| `user_fraud_scores.csv` | **Main results** | fraud_score, behavior_score, similarity_score, rating_entropy_score, cluster_score |
| `cluster_target_products.csv` | Product targeting | cluster_id, product_id, avg_cluster_fraud_score |
| `cluster_summary.csv` | Cluster metrics | num_users, avg_fraud_score, suspicious_flag |

## 🚀 Performance

### Tested on Datasets:
- ✅ Up to **1 million reviews**
- ✅ Up to **100,000+ users**
- ✅ Up to **50,000+ products**

### Complexity:
- Data loading: O(n)
- Behavioral features: O(n log n)
- Text similarity: O(n + m²) where m = unique documents
- Graph building: O(n² / 2) worst case (optimized)
- Fraud scoring: O(n)

### Memory Usage:
- TF-IDF matrix is sparse (memory-efficient)
- Graph stored efficiently with NetworkX
- Scalable to large datasets with optimization techniques

## 🔬 Fraud Detection Methodology

### Key Insights

**Behavioral Signals:**
- Abnormally high review frequency
- Rapid review bursts in short timeframes
- Concentration on specific products

**Text Signals:**
- Repetitive review text (high similarity)
- Low linguistic diversity
- Template-like patterns

**Rating Signals:**
- Unnatural rating distributions
- Always giving 5-star (or always same rating)
- No negative reviews

**Network Signals:**
- Tight clusters of co-reviewers
- Shared product targeting
- High clustering coefficient

## 💡 Example Use Cases

1. **E-commerce Platforms**: Identify and remove fake reviews
2. **Brand Protection**: Detect coordinated attack campaigns
3. **Research**: Study fraud patterns in review ecosystems
4. **Risk Assessment**: Score new users for review authenticity
5. **Investigation**: Find fraud ring members and targets

## 📝 Configuration

### Custom Fraud Score Weights

In `main.py`, modify the `weights` parameter:

```python
custom_weights = {
    'behavior': 0.4,      # Increase weight for activity patterns
    'similarity': 0.2,
    'rating': 0.2,
    'cluster': 0.2
}

results = run_pipeline(weights=custom_weights)
```

### Similarity Threshold

In `graph_analysis.py`:
```python
graph_data = build_user_graph(df, min_common_products=2)  # Require 2+ common products
```

## 🛡️ Production Considerations

- ✅ Modular architecture for easy maintenance
- ✅ Comprehensive logging for debugging
- ✅ Input validation and error handling
- ✅ Scalable to large datasets
- ✅ No data modified (read-only analysis)
- ✅ Reproducible results with proper seeding

## ⚠️ Limitations

- Text analysis limited to English reviews
- Requires sufficient review data per user for reliable patterns
- May have false positives (legitimate power reviewers flagged)
- Graph clustering effectiveness depends on product diversity
- Real-time detection would require stream processing

## 📚 Dependencies

- **pandas**: Data manipulation
- **numpy**: Numerical computing
- **scikit-learn**: ML algorithms (TF-IDF, MinMaxScaler)
- **scipy**: Statistical functions (entropy)
- **networkx**: Graph algorithms
- **streamlit**: Web dashboard
- **plotly**: Interactive visualizations

## 🔄 Workflow Example

```python
from main import run_pipeline, print_summary

# Run complete pipeline
results = run_pipeline(
    data_source='reviews_Electronics_5.json',
    output_dir='outputs',
    fraud_threshold=0.5
)

# Print summary statistics
print_summary(results)

# Access results
fraud_scores = results['fraud_scores']
suspicious_users = fraud_scores[fraud_scores['fraud_score'] > 0.7]
```

## 📞 Support

For issues, ensure:
1. All dependencies are installed: `pip install -r requirements.txt`
2. Dataset is in correct format (JSON Lines or CSV)
3. Output directory exists or can be created
4. Sufficient disk space for output files

## 📄 License

This project is provided as-is for educational and research purposes.

---

**Last Updated:** April 2026
**Maintainer:** Fraud Detection Research Team
