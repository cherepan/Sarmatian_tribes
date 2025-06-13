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


#df.describe()


with open(input_file, 'r') as f:
    print("RAW CSV CONTENT:")
    for _ in range(5):  # print first 5 lines
        print("content: ",   f.readline().strip())

print("deb 1")

ids = df.iloc[:, 0].values        # shape: (n_samples,)
X = df.iloc[:, 1:].values         # shape: (n_samples, n_features)

print("deb 2")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("deb 3")

pca = PCA(n_components=2)  # or more if needed; not needed
X_pca = pca.fit_transform(X_scaled)


print("Explained variance ratio:", pca.explained_variance_ratio_)
print("Cumulative variance explained:", np.sum(pca.explained_variance_ratio_))


print("deb 4")

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

