#!/usr/bin/env python3
import csv
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from scipy.stats import zscore
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="Berlisov_Monography_Sarmatian_burials.csv")
parser.add_argument("--config", default="plots_config.json")
parser.add_argument("--labels", default="feature_labels.json")
args = parser.parse_args()


input_file=args.csv

df = pd.read_csv(input_file, header=None)

with open("feature_labels.json", "r") as f:
    feature_labels = json.load(f)

#df.describe()


with open(input_file, 'r') as f:
    print("RAW CSV CONTENT:")
    for _ in range(5):  # print first 5 lines
        print("content: ",   f.readline().strip())



ids = df.iloc[:, 0].values        # shape: (n_samples,)
X = df.iloc[:, 5:].values         # shape: (n_samples, n_features)  starting from 5th number



scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)



pca = PCA(n_components=2)  # or more if needed; not needed
X_pca = pca.fit_transform(X_scaled)



print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Cumulative variance explained:", np.sum(pca.explained_variance_ratio_))

outliers = np.where(np.linalg.norm(X_pca, axis=1) > 3)[0]
print(" --- Outlier indices:", outliers)
print("======",X[outliers])

print("Outlier IDs:")
for idx in outliers:
    print(ids[idx])


# Assume: df is your full DataFrame, X is features only
outlier_values = X[outliers]
mean_values = X.mean(axis=0)

diff = outlier_values - mean_values  # shape (n_outliers, n_features)

# Absolute average difference across all outliers
avg_abs_diff = np.mean(np.abs(diff), axis=0)

# Sort features by strongest deviation
top_features = np.argsort(avg_abs_diff)[::-1]

# Print top 10 features
for i in top_features[:10]:
    label = feature_labels.get(str(i), f"Feature {i}")
    print(f"{label}: avg abs deviation = {avg_abs_diff[i]:.3f}")




pc1_loadings = pca.components_[0]
pc2_loadings = pca.components_[1]

# Top contributors to PC1
top_pc1 = np.argsort(np.abs(pc1_loadings))[::-1][:5]
print("Top features driving PC1:")
for i in top_pc1:
    label = feature_labels.get(str(i), f"Feature {i}")
    print(f"{label}: loading = {pc1_loadings[i]:.3f}")


    
plt.figure(figsize=(6, 5))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], 
                      c=labels if 'labels' in locals() else 'gray', 
                      cmap='tab10', s=30, alpha=0.8)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA projection (2D)")
plt.grid(True)

filename = 'PCA.png'

# Optional legend
if 'labels' in locals():
    plt.legend(*scatter.legend_elements(), title="Groups")

print("deb 5")
    
plt.tight_layout()
plt.savefig(filename, dpi=150)




