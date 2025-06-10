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


input_file  = "Berlisov_Monography_Sarmatian_burials.csv"
df = pd.read_csv(input_file, header=None)


df.describe()


#for col in df.columns:
#    print(f"Column {col}: sample values --- {df[col].unique()[:10]}")


with open("plots_config.json", "r") as f: ## here index in the .cvs file is -1 w.r.t to Berlisov's monography
    configs = json.load(f)  



for config in configs:
    condition_feature = config.get("condition_feature")
    plotting_feature = config["plotting_feature"]
    condition_value  = config.get("condition_value", None) 

    # Apply condition if present
    if condition_value is not None and condition_feature is not None:
        filtered = df[df[condition_feature] == condition_value]
        condition_text = f"Feature {condition_feature} == {condition_value}"
        filename = f"plotting_feature{plotting_feature}_if_feat{condition_feature}_eq{condition_value}.png"
    else:
        filtered = df
        condition_text = f"(All rows)"
        filename = f"plot_feat{plotting_feature}_all.png"


    if filtered.empty:
        print(f"No data for {condition_text}")
        continue


    # --- adjusted binning ---
    values = filtered[plotting_feature].dropna().unique()
    if len(values) < 20 and np.all(np.mod(values, 1) == 0):
        min_val, max_val = int(values.min()), int(values.max())
        bins = np.arange(min_val - 0.5, max_val + 1.5, 1)
    else:
        bins = 30 # arbitrary


    # --- Plot ---
    plt.figure(figsize=(6, 4))
    plt.hist(filtered[plotting_feature], bins=bins, edgecolor='black')
    plt.title(f"Feature {plotting_feature} | {condition_text}")
    plt.xlabel(f"Feature {plotting_feature}")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
