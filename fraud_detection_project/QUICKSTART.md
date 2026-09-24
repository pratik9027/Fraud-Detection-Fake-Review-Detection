# Quick Start Guide - Fraud Detection System

## 🚀 Get Started in 5 Minutes

### 1. Install Dependencies
```bash

```

### 2. Run the Pipeline (with sample data)
```bash
python main.py
```

Expected output:
```
[STEP 1] Loading and preprocessing dataset...
✓ Dataset loaded: 10000 reviews from 850 users
[STEP 2] Computing behavioral features...
✓ Behavioral features computed for 850 users
...
[FINAL] Pipeline completed successfully
```

### 3. Launch the Dashboard
```bash
streamlit run frontend/app.py
```

Open browser: `http://localhost:8501`

---

## 📊 Understand the Output

### What are Fraud Scores?

**Fraud Score Range: 0.0 - 1.0**

| Score | Risk Level | Interpretation |
|-------|-----------|-----------------|
| 0.0 - 0.3 | 🟢 Low | Normal legitimate reviewer |
| 0.3 - 0.5 | 🟡 Medium | Some suspicious patterns |
| 0.5 - 0.7 | 🟠 High | Multiple fraud indicators |
| 0.7 - 1.0 | 🔴 Critical | Strong evidence of fraud |

### Main Output: `user_fraud_scores.csv`

| Column | Meaning |
|--------|---------|
| `user_id` | Reviewer identifier |
| `fraud_score` | **Overall fraud risk (0-1)** |
| `behavior_score` | Activity pattern risk |
| `similarity_score` | Text repetition risk |
| `rating_entropy_score` | Unnatural rating pattern risk |
| `cluster_score` | Network community risk |
| `reviews_per_user` | Total reviews by the user |
| `cluster_id` | Which fraud ring (if any) |

### Example Interpretation

```
User: reviewer_12345
fraud_score: 0.82 [CRITICAL RISK]
  - behavior_score: 0.90 (very high activity, burst pattern)
  - similarity_score: 0.75 (repetitive review text)
  - rating_entropy_score: 0.88 (always 5 stars)
  - cluster_score: 0.91 (connected to fraud ring)
reviews_per_user: 147 (suspicious volume)
cluster_id: 5 (part of fraud ring #5)

=> VERDICT: Likely coordinated fraud reviewer
```

---

## 🎯 Using the Dashboard

### Tab 1: 📈 Analysis
- **Fraud Score Distribution**: See how many users fall into each risk category
- **Score Components**: Understand which factors drive fraud detection

### Tab 2: 🚨 Suspicious Users
- **Top Users**: Find the most suspicious reviewers
- **Search**: Look up specific user details
- **Download**: Export results to CSV

### Tab 3: 🔗 Clusters
- **Fraud Rings**: Identify coordinated groups
- **Cluster Metrics**: Size, composition, risk level

### Tab 4: 🎯 Products
- **Targeted Products**: Which products are attacked by fraud rings
- **Concentration**: How many reviews are on each product

---

## 📂 Generated Output Files

After running `python main.py`, check `outputs/` folder:

```
outputs/
├── clean_dataset.csv                  ← Cleaned reviews
├── user_behavior_metrics.csv          ← Activity patterns
├── user_similarity_metrics.csv        ← Text analysis
├── user_rating_metrics.csv            ← Rating patterns
├── cluster_membership.csv             ← Fraud ring assignments
├── user_graph_metrics.csv             ← Network metrics
├── user_fraud_scores.csv              ← ⭐ MAIN RESULTS
├── cluster_target_products.csv        ← Products under attack
└── cluster_summary.csv                ← Fraud ring summaries
```

---

## 💻 Using with Your Data

### Option A: JSON Format (Amazon Reviews)
```bash
python main.py path/to/reviews_Electronics_5.json
```

Expected JSON structure (one per line):
```json
{"reviewerID": "user123", "asin": "prod456", "reviewText": "Great product", "overall": 5, "unixReviewTime": 1234567890}
```

### Option B: CSV Format
```bash
python main.py data/your_reviews.csv
```

Required CSV columns:
- `reviewerID` (or `user_id`)
- `asin` (or `product_id`)
- `overall` (or `rating`)
- `unixReviewTime` (or `timestamp`)
- `reviewText` (optional but recommended)

---

## 🔍 Quick Analysis Examples

### Find all high-risk users
```bash
cd outputs
cat user_fraud_scores.csv | awk -F',' '$2 > 0.7 {print $1, $2}'
```

### Count users by risk level
```bash
python -c "
import pandas as pd
df = pd.read_csv('outputs/user_fraud_scores.csv')
print('High Risk:', (df['fraud_score'] > 0.7).sum())
print('Medium Risk:', ((df['fraud_score'] >= 0.5) & (df['fraud_score'] <= 0.7)).sum())
print('Low Risk:', (df['fraud_score'] <= 0.3).sum())
"
```

### Get top 10 suspicious users
```bash
python -c "
import pandas as pd
df = pd.read_csv('outputs/user_fraud_scores.csv')
print(df.head(10)[['user_id', 'fraud_score', 'reviews_per_user', 'cluster_id']])
"
```

---

## ⚙️ Advanced: Customizing Fraud Scores

Edit `main.py` to change component weights:

```python
# Default weights
weights = {
    'behavior': 0.3,      # Activity patterns
    'similarity': 0.25,   # Text repetition
    'rating': 0.2,        # Rating patterns
    'cluster': 0.25       # Network community
}

# Example: Emphasize behavioral signals
custom_weights = {
    'behavior': 0.5,      # INCREASED
    'similarity': 0.2,    # DECREASED
    'rating': 0.15,
    'cluster': 0.15
}

results = run_pipeline(weights=custom_weights)
```

---

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError"
```bash
# Install missing packages
pip install -r requirements.txt
```

### Error: "No such file or directory"
```bash
# Make sure you're in the right directory
cd fraud_detection_project
python main.py
```

### Error: "Empty dataset"
```bash
# Use sample data (no argument needed)
python main.py
# Or provide valid data file
python main.py reviews_Electronics_5.json
```

### Dashboard doesn't open
```bash
# Make sure you ran the pipeline first
python main.py
# Then launch dashboard
streamlit run frontend/app.py
# It opens at http://localhost:8501
```

---

## 📈 Expected Performance

| Dataset Size | Time | Memory |
|--------------|------|--------|
| 10K reviews | ~5 sec | 100 MB |
| 100K reviews | ~30 sec | 500 MB |
| 1M reviews | ~5 min | 2-3 GB |

---

## 💡 Key Metrics Explained

### Fraud Score Components

1. **Behavior Score** (30% weight)
   - How fast they review (reviews per day)
   - Concentrated activity in time windows (burst score)
   - Focusing on specific products

2. **Similarity Score** (25% weight)
   - How similar their reviews are to each other
   - Do they copy and paste text?
   - Or do they write unique reviews each time?

3. **Rating Entropy Score** (20% weight)
   - Do they give diverse ratings (1-5 stars)?
   - Or always the same rating (e.g., always 5)?
   - Real users vary; fraudsters don't

4. **Cluster Score** (25% weight)
   - Are they connected to other suspicious reviewers?
   - Do they form tight groups (fraud rings)?
   - Network analysis captures coordinated behavior

---

## 🎓 Understanding Fraud Patterns

### Classic Fraud Indicators

✅ **These trigger HIGH fraud scores:**

- **Behavioral**: Suddenly 100 reviews in 2 days
- **Textual**: Every review says "Love it! Best buy ever!"
- **Ratings**: All 5 stars on products, all 1 star on competitors
- **Network**: 50 users all reviewing same products together
- **Combined**: Everything above + tight-knit group = likely fraud ring

✅ **These trigger LOW fraud scores:**

- **Behavioral**: 5-10 reviews over 1-2 years
- **Textual**: Each review unique, detailed, personal
- **Ratings**: Mix of ratings (3, 4, 5 stars)
- **Network**: Independent from other reviewers

---

## 📞 Next Steps

1. ✅ Run pipeline: `python main.py`
2. ✅ View results: Open `outputs/user_fraud_scores.csv`
3. ✅ Explore dashboard: `streamlit run frontend/app.py`
4. ✅ Use with your data: `python main.py your_data.json`
5. ✅ Export findings: Download from dashboard

---

**Happy Fraud Hunting! 🔍**
