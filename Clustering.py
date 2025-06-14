#!/usr/bin/env python3
import csv
import os
import pandas as pd
import numpy as np
import json

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import OPTICS
from sklearn.cluster import Birch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation



import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="DataSets1234.csv")
parser.add_argument("--config", default="plots_config.json")
parser.add_argument("--labels", default="feature_labels.json")
args = parser.parse_args()


def plot_clusters(X_plot, labels, title):
    plt.figure(figsize=(6,5))
    sns.scatterplot(x=X_plot[:, 0], y=X_plot[:, 1], hue=labels, palette='tab10', s=20)
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title='Cluster')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(title, dpi=150)




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

kmeans = KMeans(n_clusters=4, random_state=0)
labels_kmeans = kmeans.fit_predict(X_scaled)
plot_clusters(X_pca, labels_kmeans, "KMeans")

db = DBSCAN(eps=1.5, min_samples=5)
labels_db = db.fit_predict(X_scaled)
plot_clusters(X_pca, labels_db, "DBSCAN")

agg = AgglomerativeClustering(n_clusters=4)
labels_agg = agg.fit_predict(X_scaled)
plot_clusters(X_pca, labels_agg, "AgglomerativeClustering")

spec = SpectralClustering(n_clusters=4, affinity='nearest_neighbors', assign_labels='kmeans', random_state=0)
labels_spec = spec.fit_predict(X_scaled)
plot_clusters(X_pca, labels_spec, "SpectralClustering")


gmm = GaussianMixture(n_components=4, random_state=0)
labels_gmm = gmm.fit_predict(X_scaled)
plot_clusters(X_pca, labels_gmm, "GaussianMixture")

labels = OPTICS(min_samples=5, xi=0.05).fit_predict(X_scaled)
plot_clusters(X_pca, labels, "OPTICS")


labels = Birch(n_clusters=4).fit_predict(X_scaled)
plot_clusters(X_pca, labels, "BIRCH")


labels = AgglomerativeClustering(n_clusters=4, linkage='ward').fit_predict(X_scaled)
plot_clusters(X_pca, labels, "WardHierarchicalClustering")

labels = AffinityPropagation().fit_predict(X_scaled)
plot_clusters(X_pca, labels, "AffinityPropagation")
