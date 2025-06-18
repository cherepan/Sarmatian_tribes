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
from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import csv
import os

from libs.archaeology_analyzer import ArchaeologyAnalyzer



def run_classification(analyzer, models_to_run):
    models_to_run = [m.lower() for m in models_to_run]

    if "logistic" in models_to_run or "all" in models_to_run:
        analyzer.run_classifier(LogisticRegression(max_iter=1000), "Logistic Regression", scale=True)

    if "randomforest" in models_to_run or "all" in models_to_run:
        analyzer.run_classifier(RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest")

    if "gradientboosting" in models_to_run or "all" in models_to_run:
        analyzer.run_classifier(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42), "Gradient Boosting")

    if "svm" in models_to_run or "all" in models_to_run:
        analyzer.run_classifier(SVC(kernel='rbf', C=1.0, gamma='scale'), "SVM", scale=True)

    if "xgboost" in models_to_run or "all" in models_to_run:
        analyzer.encode_labels()
        analyzer.run_classifier(
            XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
            "XGBoost"
        )



def inspect_random_sample(analyzer, model):
    print(f" Random Sample Prediction: model {model}  ")
    X, ids = analyzer.X, analyzer.ids
    y = analyzer.y

    idx = np.random.randint(len(X))
    sample = X[idx].reshape(1, -1)

    try:
        sample_scaled = analyzer.scaler.transform(sample)
    except:
        print("Scaler not fitted - using raw sample")
        sample_scaled = sample


    predicted = model.predict(sample_scaled)[0]
    print(f"Sample ID: {ids[idx]}")
    print(f"True label: {y[idx]}")
    print(f"Predicted: {predicted}")



def compare_models(analyzer, models_to_run):
    results = []
    models_to_run = [m.lower() for m in models_to_run]
    classifiers = {
        "logistic": LogisticRegression(max_iter=1000),
        "randomforest": RandomForestClassifier(n_estimators=100, random_state=42),
        "gradientboosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
        "svm": SVC(kernel='rbf', C=1.0, gamma='scale', probability=True),
        "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }
    for name, model in classifiers.items():
        if name in models_to_run or "all" in models_to_run:
            analyzer.encode_labels() if name == "xgboost" else None
            X_train, X_test, y_train, y_test = analyzer.split_data()

            if name in ["logistic", "svm"]:
                X_train = analyzer.scaler.fit_transform(X_train)
                X_test = analyzer.scaler.transform(X_test)

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            prec = precision_score(y_test, y_pred, average='weighted')
            rec = recall_score(y_test, y_pred, average='weighted')

            results.append({
                "Model": name,
                "Accuracy": acc,
                "F1": f1,
                "Precision": prec,
                "Recall": rec
            })
            analyzer.models[name] = model  # Save model

    df_results = pd.DataFrame(results).sort_values(by="F1", ascending=False)
    print("\n Model Comparison:\n", df_results.to_string(index=False))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="DataSets1234.csv")
    parser.add_argument("--config", default="plots_config.json")
    parser.add_argument("--labels", default="feature_labels.json")
    args = parser.parse_args()

    analyzer = ArchaeologyAnalyzer(args.csv, args.labels, args.config)



    analyzer.prepare_data(drop_feature_value=11, feature_idx=3, target_column=4, exclude_feature_indices=[1,2,3,4])

    models = ['randomforest']
    run_classification(analyzer, models)
    compare_models(analyzer,models)

#    # --- Inspect one prediction manually ---
#    if "XGBoost" in analyzer.models:
#        inspect_random_sample(analyzer, analyzer.models["XGBoost"])
                                        

    # --- Clustering ---
    pca, X_pca =  analyzer.run_clustering(method="kmeans", n_clusters=5)
#    pca, X_pca =  analyzer.run_clustering(method="minibatchkmeans", n_clusters=5)
#    pca, X_pca =  analyzer.run_clustering(method="dbscan", n_clusters=5)
#    pca, X_pca =  analyzer.run_clustering(method="optics", n_clusters=5)
#    pca, X_pca =  analyzer.run_clustering(method="agglomerative", n_clusters=5)
#    pca, X_pca =  analyzer.run_clustering(method="spectral", n_clusters=5)
#    pca, X_pca =  analyzer.run_clustering(method="affinity", n_clusters=5)
#    pca, X_pca =  analyzer.run_clustering(method="gmm", n_clusters=5)
#    pca, X_pca =  analyzer.run_clustering(method="bayesian", n_clusters=5)


    analyzer.explain_pca_clusters(n_clusters=2, top_n=50, output_dir="plots")
    analyzer.compare_clusters_on_feature(10)
    analyzer.compare_clusters_on_feature(107)
    analyzer.compare_clusters_on_feature(22)

#    analyzer.print_pca_feature_contributions(
#    components=pca.components_,
#    feature_names=analyzer.feature_labels
#    )

#    analyzer.print_pca_feature_contributions(pca_components=pca.components_, analyzer.feature_labels)

    analyzer.plot_pca_feature_contributions(pca.components_, analyzer.feature_labels, top_n=15)
    analyzer.run_factor_analysis(n_factors=10)
#    analyzer.plot_feature_evolution(9)
    # --- Factor Analysis
#    for i in range(5,100):
#        analyzer.plot_feature_evolution(i)
    
if __name__ == "__main__":
    main()
