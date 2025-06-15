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
from sklearn.decomposition import FactorAnalysis


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

    def prepare_data(self, drop_feature_value=None, feature_idx=3, target_column=3):
        """
        Prepares X, y, and ids from the raw DataFrame.
        - drop_feature_value: optional value to drop from feature_idx column.
        - feature_idx: used only for filtering rows (e.g., remove Dating=11).
        - target_column: the column used for supervised learning (e.g., 3 or 4).
        """
        if drop_feature_value is not None:
            self.df = self.df[self.df.iloc[:, feature_idx] != drop_feature_value]
    
        self.ids = self.df.iloc[:, 0].values
        self.X = self.df.iloc[:, 5:].values  # starting from 5th index
        self.y = self.df.iloc[:, target_column].values
        self.target_column = target_column
        print(f"[INFO] Prepared data: X shape {self.X.shape}, target column = {target_column}")

        
        
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


    def run_factor_analysis(self, n_factors=5, max_features=100):
        """
        Perform Factor Analysis and plot the factor loadings heatmap.
        Args:
        n_factors (int): Number of latent factors to extract.
        max_features (int): Max number of features to show for readability.
        """


        # Ensure data is scaled
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Factor Analysis
        fa = FactorAnalysis(n_components=n_factors, random_state=42)
        factors = fa.fit_transform(X_scaled)
        loadings = fa.components_.T  # shape: (n_features, n_factors)


        feature_labels = self.feature_labels
        feature_names = [self.feature_labels.get(str(i + 5), f"Feature {i+5}") for i in range(loadings.shape[0])]


        # Create DataFrame with loadings
        loading_df = pd.DataFrame(
            loadings[:max_features],
            index=feature_names[:max_features],
            columns=[f"Factor {i+1}" for i in range(n_factors)]
        )



        # Plot heatmap
        plt.figure(figsize=(12, 0.5 * max_features))
        sns.heatmap(loading_df, annot=True, cmap="coolwarm", center=0, fmt=".2f")
        plt.title(f"Factor Loadings (first {max_features} features)", fontsize=12)
        plt.tight_layout()
        plt.savefig(f"factor_analys_with_{max_features}_features_{n_factors}_factors.png")
        plt.tight_layout()

