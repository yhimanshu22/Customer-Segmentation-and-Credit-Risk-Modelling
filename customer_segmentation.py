#!/usr/bin/env python
# coding: utf-8

# # Customer Segmentation using K-Means Clustering & PCA
# 
# ## Project Overview
# This script implements customer segmentation based on credit card usage behavior. 
# We use **K-Means Clustering** to group customers and **PCA (Principal Component Analysis)** for dimensionality reduction and visualization.
# 
# ### Objectives:
# 1.  **Data Preprocessing**: Handle missing values and scale features.
# 2.  **Dimensionality Reduction**: Use PCA to reduce noise and visualize high-dimensional data.
# 3.  **Clustering**: Apply K-Means to segment customers.
# 4.  **Analysis**: Interpret the characteristics of each segment.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Set plot style
sns.set(style="whitegrid")
plt.style.use('fivethirtyeight')

# ## 1. Data Loading
# We load the `CC GENERAL.csv` dataset downloaded from Kaggle.

try:
    df = pd.read_csv('CC GENERAL.csv')
    print("Dataset loaded successfully.")
    print(df.head())
except FileNotFoundError:
    print("Error: 'CC GENERAL.csv' not found. Please run 'download_data.py' to fetch the dataset.")
    exit()

# ## 2. Data Preprocessing
# 
# ### Handling Missing Values
# We check for missing values. `MINIMUM_PAYMENTS` and `CREDIT_LIMIT` often have missing values. We'll fill them with the median to be robust against outliers.

# Check for nulls
print("\nMissing values before imputation:")
print(df.isnull().sum())

# Fill missing values with median
df['MINIMUM_PAYMENTS'].fillna(df['MINIMUM_PAYMENTS'].median(), inplace=True)
df['CREDIT_LIMIT'].fillna(df['CREDIT_LIMIT'].median(), inplace=True)

# Drop CUST_ID as it's not a feature
if 'CUST_ID' in df.columns:
    df.drop('CUST_ID', axis=1, inplace=True)

print("\nMissing values after imputation:")
print(df.isnull().sum())

# ### Feature Scaling
# **Why Scale?** K-Means is distance-based (Euclidean distance). Features with larger ranges (e.g., `BALANCE` vs `PURCHASES_FREQUENCY`) would dominate the distance calculation if not scaled. We use `StandardScaler` to transform features to mean=0 and std=1.

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Convert back to DataFrame for readability
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
print("\nScaled Data Head:")
print(df_scaled.head())

# ## 3. Dimensionality Reduction (PCA)
# 
# **Why PCA?**
# 1.  **Noise Reduction**: Removes correlated features and focuses on maximum variance.
# 2.  **Visualization**: Allows us to visualize high-dimensional data in 2D or 3D.
# 
# We will reduce the data to 2 components for visualization.

pca = PCA(n_components=2)
principal_components = pca.fit_transform(df_scaled)
pca_df = pd.DataFrame(data=principal_components, columns=['PCA1', 'PCA2'])

print(f"\nExplained Variance Ratio: {pca.explained_variance_ratio_}")
print(pca_df.head())

# ## 4. K-Means Clustering
# 
# ### Determining Optimal K (Elbow Method)
# We use the **Elbow Method** to find the optimal number of clusters. We plot the Within-Cluster Sum of Squares (WCSS) against the number of clusters. The "elbow" point represents a good balance between compactness and number of clusters.

wcss = []
range_n_clusters = range(1, 11)

for i in range_n_clusters:
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.savefig('images/elbow_method.png')
plt.close()
print("\nSaved 'images/elbow_method.png'")

# ### Applying K-Means
# Based on the Elbow plot (usually around k=4 or k=5), we apply K-Means. Let's choose **k=4** for this example.

k = 4  # Chosen based on Elbow method
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(df_scaled)

# Add cluster labels to original and PCA dataframes
df['Cluster'] = clusters
pca_df['Cluster'] = clusters

# ## 5. Visualization & Analysis
# 
# We visualize the clusters using the 2 PCA components.

plt.figure(figsize=(10, 8))
sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=pca_df, palette='viridis', s=100, alpha=0.8)
plt.title('Customer Segments (PCA Visualization)')
plt.savefig('images/customer_segments_pca.png')
plt.close()
print("Saved 'images/customer_segments_pca.png'")

# ### Cluster Profiling
# Let's look at the mean values of key features for each cluster to understand their behavior.

cluster_summary = df.groupby('Cluster')[['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS']].mean()
print("\nCluster Summary:")
print(cluster_summary)

# **Interpretation (Example):**
# - **Cluster 0**: Low balance, low purchases. (Inactive/Low value)
# - **Cluster 1**: High balance, high cash advance, low purchases. (High risk/Cash users)
# - **Cluster 2**: High purchases, high payments. (Premium/Transactors)
# - **Cluster 3**: Moderate balance, moderate purchases. (Average users)
# 
# *Note: Actual interpretation depends on the run results.*

# ## Conclusion
# - **Dimensionality Reduction**: PCA explains ~47% of variance with 2 components.
# - **Clustering**: K-Means identified distinct customer segments (e.g., based on Balance and Purchases).
# - **Next Steps**: Analyze cluster centroids to label segments (e.g., 'Big Spenders', 'Transactors').
