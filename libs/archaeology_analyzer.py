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



class ArchaeologyAnalyzer:
    def __init__(self, csv_path, labels_path, config_path=None):
        self.df = pd.read_csv(csv_path, header=None)
        self.feature_labels = json.load(open(labels_path))
        self.configs = json.load(open(config_path)) if config_path else None
        self.models = {}
        self.X = None
        self.y = None
        self.ids = None
        self.scaler = StandardScaler()
        self.le = LabelEncoder()


    def prepare_data(self, drop_feature_value=None, feature_idx=3):
        if drop_feature_value is not None:
            self.df = self.df[self.df.iloc[:, feature_idx] != drop_feature_value]
        self.ids = self.df.iloc[:, 0].values
        self.X = self.df.iloc[:, 5:].values
        self.y = self.df.iloc[:, feature_idx].values


        
    def show_feature_values(self, idx):
        unique_vals = sorted(self.df.iloc[:, idx].dropna().unique())
        print(f"Feature {idx}: possible values {unique_vals}")


        
    def split_data(self):
        return train_test_split(self.X, self.y, stratify=self.y, random_state=42)


    
    def run_classifier(self, model, model_name, scale=False):
        X_train, X_test, y_train, y_test = self.split_data()
        if scale:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f"\n{model_name}\n", classification_report(y_test, y_pred))
        self.models[model_name] = model


        # Feature importance (if available)
        if hasattr(model, "feature_importances_"):
            self.plot_feature_importance(model.feature_importances_, model_name)
        elif model_name == "Logistic Regression":
            self.plot_feature_importance(np.abs(model.coef_).mean(axis=0), model_name)
        else:
            try:
                result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
                self.plot_feature_importance(result.importances_mean, model_name)
            except Exception as e:
                print(f"Could not compute feature importance for {model_name}: {e}")


    def plot_feature_importance(self, importances, model_name):
        indices = np.argsort(importances)[::-1][:35]  # top 15
        labels = [self.feature_labels.get(str(i+5), f"Feature {i+5}") for i in indices]  # +5 offset for feature columns
        plt.figure(figsize=(8, 6))
        sns.barplot(x=importances[indices], y=labels, orient="h")
        plt.title(f"Top Feature Importances: {model_name}")
        plt.tight_layout()
        plt.savefig(f"feature_importance_{model_name.replace(' ', '_')}.png")
        plt.close()

    def encode_labels(self):
        self.y = self.le.fit_transform(self.y)



    def plot_selected_features(self):
        if not self.configs:
            print("No config file provided for plotting.")
            return
        os.makedirs("plots", exist_ok=True)

        for config in self.configs:
            condition_feature = config.get("condition_feature")
            plotting_feature = config["plotting_feature"]
            condition_value = config.get("condition_value", None)

            if condition_value is not None and condition_feature is not None:
                filtered = self.df[self.df[condition_feature] == condition_value]
                cond_label = self.feature_labels.get(str(condition_feature), f"Feature {condition_feature}")
                condition_text = f"{cond_label} (feature {condition_feature}) == {condition_value}"
                filename = f"plot_feat{plotting_feature}_if_feat{condition_feature}_eq{condition_value}.png"
            else:
                filtered = self.df
                condition_text = "(All rows)"
                filename = f"plot_feat{plotting_feature}_all.png"

            if filtered.empty:
                print(f"No data for {condition_text}")
                continue

            label_name = self.feature_labels.get(str(plotting_feature), f"Feature {plotting_feature}")
            label_full = f"{label_name} (feature {plotting_feature})"
            values = filtered[plotting_feature].dropna().unique()

            if len(values) < 20 and np.all(np.mod(values, 1) == 0):
                min_val, max_val = int(values.min()), int(values.max())
                bins = np.arange(min_val - 0.5, max_val + 1.5, 1)
            else:
                bins = 30

            plt.figure(figsize=(6, 4))
            plt.hist(filtered[plotting_feature], bins=bins, edgecolor='black')
            plt.title(f"{label_full} | {condition_text}", fontsize=9)
            plt.xlabel(label_full)
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join("plots", filename), dpi=150)
            plt.close()
            print(f"Saved plot: {filename}")        



    def run_clustering(self, method="kmeans", n_clusters=3):
        X_scaled = self.scaler.fit_transform(self.X)
        if method == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == "dbscan":
            model = DBSCAN(eps=0.5, min_samples=5)
        elif method == "agglomerative":
            model = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            raise ValueError("Unsupported clustering method")

        labels = model.fit_predict(X_scaled)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        plt.figure(figsize=(6, 5))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=10)
        plt.title(f"Clustering: {method} (PCA projection)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(f"clustering_{method}.png")
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="DataBase_3.csv")
    parser.add_argument("--config", default="plots_config.json")
    parser.add_argument("--labels", default="feature_labels.json")
    
    args = parser.parse_args()

    analyzer = ArchaeologyAnalyzer(args.csv, args.labels, args.config)
    analyzer.prepare_data(drop_feature_value=11)
    analyzer.show_feature_values(3)

    analyzer.run_classifier(LogisticRegression(max_iter=1000), "Logistic Regression", scale=True)
    analyzer.run_classifier(RandomForestClassifier(n_estimators=100, random_state=42), "Random Forest")
    analyzer.run_classifier(GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42), "Gradient Boosting")
    analyzer.run_classifier(SVC(kernel='rbf', C=1.0, gamma='scale'), "SVM", scale=True)

    analyzer.encode_labels()
    analyzer.run_classifier(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42), "XGBoost")

    analyzer.plot_selected_features()
    analyzer.run_clustering(method="kmeans", n_clusters=5)
    analyzer.run_clustering(method="dbscan")
