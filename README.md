# Week 4: Exploratory Data Analysis and Feature Engineering (Beauty & Wellness Market)

**Data Science with Python Analyst Internship** | Yuva Intern (Remote, Coimbatore)
**Intern:** Asad Khan

## Overview

This repository holds my Week 4 task: a detailed plan for **Exploratory Data Analysis (EDA)** and **Feature Engineering** on preprocessed data from the **beauty & wellness market**.

The task is a planning and documentation exercise. No dataset was supplied, so the work assumes a typical transaction-level dataset from a beauty & wellness retailer (online store plus salon and spa services). All findings and numbers in the document are **hypothetical examples**, used to show how the approach would work on real data.

## Deliverable

The full report is in this repository: [`Week4_EDA_18_Points.docx`](./Week4_EDA_18_Points.docx). It covers all 18 points of the task.

## Assumed Dataset

| Column | Type | Description |
|---|---|---|
| customer_id | Identifier | Unique customer code |
| order_id, order_date | ID / Date | Transaction identifier and purchase date |
| product_category | Categorical | Skincare, haircare, makeup, fragrance, wellness, salon service |
| brand_tier | Categorical | Budget, mid-range, premium |
| unit_price, quantity, discount_pct | Numeric | Price paid, units bought, discount applied |
| channel | Categorical | Website, mobile app, in-store, marketplace |
| city, age, gender | Mixed | Customer demographics and location |
| rating | Numeric (1-5) | Customer review score |
| is_repeat | Binary | Whether the customer bought again within 90 days |

## What the Document Covers

### 1. EDA workflow (10 steps)
Define business questions, data overview, quality re-check, univariate, bivariate and multivariate analysis, time-series analysis, customer segmentation, hypothesis testing, and an insight summary.

### 2. Statistical methods
Descriptive statistics, skewness and kurtosis, Pearson and Spearman correlation, independent t-test, ANOVA, chi-square test, Shapiro-Wilk normality test, IQR and Z-score outlier detection.

### 3. Visualizations
Histogram with KDE, box and violin plots, count and bar plots, line plots, scatter plots, correlation heatmap, pair plot, category-by-month pivot heatmap, and cohort retention heatmap.

### 4. Basic and advanced EDA techniques
- **Basic:** summary statistics, frequency analysis, distribution checks, outlier detection, correlation, group-by aggregation.
- **Advanced:** multivariate heatmaps, cohort analysis, RFM analysis, PCA, K-Means clustering, VIF multicollinearity check, market basket analysis.

### 5. Feature engineering
- **New variables:** `order_value`, `recency_days`, `frequency`, `monetary_value`, `avg_days_between_orders`, `is_festive_season`, `is_weekend`, `discount_sensitivity`, `category_diversity`, `premium_share`, `age_group`.
- **Transformations:** scaling (Min-Max and standardization), log transform, one-hot and ordinal encoding, binning, date decomposition, interaction variables (for example `discount x brand_tier`).

### 6. Hidden behaviour and market signals
Replenishment habit, deal dependence, trend adoption, cross-category journey, loyalty drift, seasonal demand, review sentiment drift, and channel preference.

### 7. Business scenario
A hypothetical skincare brand, **GlowRoots**, has a low 90-day repeat purchase rate. The document shows how engineered features (reorder cycle, discount sensitivity, category diversity, RFM segments, rolling rating) lead to concrete decisions: targeted discounts instead of blanket coupons, timed reorder reminders, cross-category nudges, a win-back campaign, and a repeat-purchase prediction model.

## Python Libraries Planned

| Library | Role |
|---|---|
| pandas | Loading data, validation, group-by, pivot tables, date handling, feature creation |
| NumPy | Numerical operations, log transforms, `np.where`, percentiles |
| matplotlib | Base plotting: layouts, line charts, subplots, labels |
| seaborn | Distributions, box plots, heatmaps, pair plots |
| SciPy (`scipy.stats`) | Hypothesis tests, correlation tests, normality tests |
| scikit-learn | Scaling, encoding, PCA, K-Means |

## Sample Code

Starter snippet from the document (loading and validating the data, then creating core features):

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")
df = pd.read_csv("beauty_wellness_clean.csv", parse_dates=["order_date"])

# Validation of ranges
assert (df["unit_price"] > 0).all()
assert df["rating"].between(1, 5).all()

# Core derived columns
df["revenue"] = df["unit_price"] * df["quantity"] * (1 - df["discount_pct"] / 100)
df["month"] = df["order_date"].dt.month
df["is_weekend"] = np.where(df["order_date"].dt.dayofweek >= 5, 1, 0)
```

RFM feature example:

```python
snapshot = df["order_date"].max() + pd.Timedelta(days=1)
rfm = df.groupby("customer_id").agg(
    recency_days=("order_date", lambda x: (snapshot - x.max()).days),
    frequency=("order_id", "nunique"),
    monetary_value=("revenue", "sum"),
)
rfm["R"] = pd.qcut(rfm["recency_days"], 4, labels=[4, 3, 2, 1])
rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4])
rfm["M"] = pd.qcut(rfm["monetary_value"], 4, labels=[1, 2, 3, 4])
rfm["rfm_score"] = rfm[["R", "F", "M"]].astype(int).sum(axis=1)
```

## Repository Structure

```
.
├── README.md
└── Week4_EDA_18_Points.docx
```

## Time Spent

About 33 hours, against an intended workload of 30-35 hours.

## Author

**Asad Khan**
BTech CSE (Final Year), Integral University, Lucknow
GitHub: [asadkhan8757](https://github.com/asadkhan8757)
