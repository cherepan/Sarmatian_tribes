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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier
from sklearn.svm import SVC
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="DataSets1234.csv")
parser.add_argument("--config", default="plots_config.json")
parser.add_argument("--labels", default="feature_labels.json")
args = parser.parse_args()

input_file=args.csv

df = pd.read_csv(input_file, header=None)
df = df[df.iloc[:, 3] != 11] # Drop rows where feature 3 == 11; Dating  - VII century

with open("feature_labels.json", "r") as f:
    feature_labels = json.load(f)

#df.describe()


with open(input_file, 'r') as f:
    print("RAW CSV CONTENT:")
    for _ in range(5):  # print first 5 lines
        print("content: ",   f.readline().strip())



ids = df.iloc[:, 0].values        # shape: (n_samples,)
X = df.iloc[:, 5:].values         # shape: (n_samples, n_features)  starting from 5th number

y = df.iloc[:, 3].values          # Raw Dating
#y = df.iloc[:, 4].values          # Culture label (column 2)

feature_idx = 3  # for example
all_values = df.iloc[:, feature_idx].dropna().unique()
print(f"Feature {feature_idx} values:", all_values)


counts = df.iloc[:, 3].value_counts().sort_index()
print("Sample count per value of feature 3:\n", counts)








X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Standardize (important for some models like SVM & LogisticRegression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

print(" Logistic Regression\n", classification_report(y_test, y_pred))





model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(" Random Forest\n", classification_report(y_test, y_pred))


model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("  Gradient Boosting\n", classification_report(y_test, y_pred))




model = SVC(kernel='rbf', C=1.0, gamma='scale')
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
print(" SVM\n", classification_report(y_test, y_pred))


le = LabelEncoder()
y_encoded = le.fit_transform(y)  # Now y_encoded is 0-based integers
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, stratify=y_encoded, random_state=42)




model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("XGBoost\n", classification_report(y_test, y_pred))
