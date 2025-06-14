#!/usr/bin/env python3
import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.inspection import permutation_importance
import csv
import os

from libs.archaeology_analyzer import ArchaeologyAnalyzer





parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="DataBase_3.csv")
parser.add_argument("--config", default="plots_config.json")
parser.add_argument("--labels", default="feature_labels.json")

args = parser.parse_args()

analyzer = ArchaeologyAnalyzer(args.csv, args.labels, args.config)
analyzer.prepare_data(drop_feature_value=11)
analyzer.show_feature_values(3)

#analyzer.run_classifier(LogisticRegression(max_iter=1000), "Logistic Regression", scale=True)
#analyzer.run_classifier(RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest")
#analyzer.run_classifier(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42), "Gradient Boosting")
#analyzer.run_classifier(SVC(kernel='rbf', C=1.0, gamma='scale'), "SVM", scale=True)

analyzer.encode_labels()
analyzer.run_classifier(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42), "XGBoost")

#analyzer.plot_selected_features()
analyzer.run_clustering(method="kmeans", n_clusters=5)
#analyzer.run_clustering(method="dbscan")
