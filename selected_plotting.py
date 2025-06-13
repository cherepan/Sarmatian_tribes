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


df = pd.read_csv(args.csv, header=None)


df.describe()



with open(args.config, "r") as f: ## here index in the .cvs file is -1 w.r.t to Berlisov's monography
    configs = json.load(f)  

with open(args.labels, "r") as f:
    feature_labels = json.load(f)

for col in df.columns:
    unique_vals = sorted(df[col].dropna().unique())
    print(f"Feature {col}: possible values {unique_vals}")
    
print("deb 1")
for config in configs:
    condition_feature = config.get("condition_feature")
    plotting_feature = config["plotting_feature"]
    condition_value  = config.get("condition_value", None) 

    # Apply the condition if present
    if condition_value is not None and condition_feature is not None:
        filtered = df[df[condition_feature] == condition_value]
        cond_label = feature_labels.get(str(condition_feature), f"Feature {condition_feature}")
        cond_label_full = f"{cond_label} (feature {condition_feature})"
        condition_text = f"{cond_label_full} == {condition_value}"
        
        filename = (
            f"plot_feat{plotting_feature}_if_feat{condition_feature}_eq{condition_value}.png"
        )
        
#        condition_text = f"Feature {condition_feature} == {condition_value}"
#        filename = f"plotting_feature{plotting_feature}_if_feat{condition_feature}_eq{condition_value}.png"
    else:
        filtered = df
        condition_text = f"(All rows)"
        filename = f"plot_feat{plotting_feature}_all.png"
    print("deb 2")

    if filtered.empty:
        print(f"No data for {condition_text}")
        continue
    print("deb 3")

    # --- Determine label ---
    label_name = feature_labels.get(str(plotting_feature), f"Feature {plotting_feature}")
    label_full = f"{label_name} (feature {plotting_feature})"

    print("deb 4")
    
    # --- adjusted binning ---
    values = filtered[plotting_feature].dropna().unique()
    if len(values) < 20 and np.all(np.mod(values, 1) == 0):
        min_val, max_val = int(values.min()), int(values.max())
        bins = np.arange(min_val - 0.5, max_val + 1.5, 1)
    else:
        bins = 30 # arbitrary
    print("deb 5")
    # --- Plot ---        
    os.makedirs("plots", exist_ok=True)
    filename = os.path.join("plots", filename)
    plt.figure(figsize=(6, 4))
    plt.hist(filtered[plotting_feature], bins=bins, edgecolor='black')
    plt.title(f"{label_full} | {condition_text}", fontsize=9)
    print('--------------  ', label_full)
    plt.xlabel(label_full)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

