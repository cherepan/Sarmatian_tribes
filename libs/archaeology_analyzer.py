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
from sklearn.inspection import permutation_importance
from sklearn.decomposition import FactorAnalysis
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.cluster import (KMeans, MiniBatchKMeans, DBSCAN, OPTICS,
                             AgglomerativeClustering, SpectralClustering,
                             AffinityPropagation)
from scipy.stats import ttest_ind


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
        # Optional data containers
        self.clustered_df = None        # will store PCA+cluster-labeled data
        self.X_pca = None               # optional: PCA-transformed features
        self.pca_model = None           # optional: PCA object itself
        self.cluster_labels = None      # optional: array of cluster assignments

    def prepare_data(self, drop_feature_value=None, feature_idx=3, target_column=3, exclude_feature_indices=None):
        """
        Prepares X, y, and ids from the raw DataFrame.
        - drop_feature_value: optional value to drop from feature_idx column.
        - feature_idx: used only for filtering rows (e.g., remove Dating=11).
        - target_column: the column used for supervised learning (e.g., 3 or 4).
        - exclude_feature_indices: list of column indices to exclude from X.
        """
        if drop_feature_value is not None:
            self.df = self.df[self.df.iloc[:, feature_idx] != drop_feature_value]

        self.ids = self.df.iloc[:, 0].values
        all_feature_indices = list(range(5, self.df.shape[1]))  # all features starting from column 5

        if exclude_feature_indices:
            feature_cols = [i for i in all_feature_indices if i not in exclude_feature_indices]
        else:
            feature_cols = all_feature_indices

        self.X = self.df.iloc[:, feature_cols].values
        self.y = self.df.iloc[:, target_column].values
        self.target_column = target_column

        print(f"[INFO] Prepared data: X shape {self.X.shape}, target column = {target_column}")
        if exclude_feature_indices:
            print(f"[INFO] Excluded features: {exclude_feature_indices}")





        
        
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


    def run_clustering(self, method="kmeans", n_clusters=5):
        """
        Performs clustering using the specified method.
        Supported methods: kmeans, minibatchkmeans, dbscan, optics,
        agglomerative, spectral, affinity, gmm, bayesian.
        """


        X_scaled = self.scaler.fit_transform(self.X)

        # Select clustering model
        if method == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=42)
        elif method == "minibatchkmeans":
            model = MiniBatchKMeans(n_clusters=n_clusters, random_state=42)
        elif method == "dbscan":
            model = DBSCAN(eps=0.5, min_samples=5)
        elif method == "optics":
            model = OPTICS(min_samples=5, xi=0.05)
        elif method == "agglomerative":
            model = AgglomerativeClustering(n_clusters=n_clusters)
        elif method == "spectral":
            model = SpectralClustering(n_clusters=n_clusters, assign_labels="kmeans", random_state=42)
        elif method == "affinity":
            model = AffinityPropagation(random_state=42)
        elif method == "gmm":
            model = GaussianMixture(n_components=n_clusters, random_state=42)
        elif method == "bayesian":
            model = BayesianGaussianMixture(n_components=n_clusters, random_state=42)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")

        # Compute cluster labels
        if method in {"gmm", "bayesian"}:
            labels = model.fit_predict(X_scaled)
        else:
            labels = model.fit_predict(X_scaled)

        # Perform PCA projection to 2D
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # Visualize clustering result in PCA space
        plt.figure(figsize=(6, 5))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='tab10', s=10)
        plt.title(f"Clustering: {method} (PCA projection)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig(f"clustering_{method}.png")
        plt.close()
        
        # Save results for further use
        self.X_pca = X_pca
        self.pca_model = pca
        self.cluster_labels = labels
        return pca, X_pca

    def explain_pca_clusters(self, n_clusters=2, top_n=20, output_dir="plots"):
        """
        Clusters the PCA-projected space and identifies which features distinguish the resulting clusters.
        Does not modify self.df; creates self.clustered_df instead.
        """
        os.makedirs(output_dir, exist_ok=True)

        # 1. PCA transformation
        X_scaled = self.scaler.fit_transform(self.X)
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # 2. Clustering
        cluster_labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(X_pca)
        self.clustered_df = self.df.copy()
        self.clustered_df["pca_cluster"] = cluster_labels

        # 3. Compare features
        results = []
        for i in range(self.X.shape[1]):
            col_idx = i + 5
            name = self.feature_labels.get(str(col_idx), f"Feature {col_idx}")
            label_with_index = f"{name} ( {col_idx} )"
            
            values0 = self.clustered_df[self.clustered_df["pca_cluster"] == 0].iloc[:, col_idx].dropna()
            values1 = self.clustered_df[self.clustered_df["pca_cluster"] == 1].iloc[:, col_idx].dropna()

            
            if len(values0) > 10 and len(values1) > 10:
                mean0 = values0.mean()
                mean1 = values1.mean()
                diff = abs(mean0 - mean1)
                results.append((label_with_index, mean0, mean1, diff))


        # 4. Export table
        df_diff = pd.DataFrame(results, columns=["Feature", "Cluster 0 Mean", "Cluster 1 Mean", "Abs. Difference"])
        df_diff.sort_values(by="Abs. Difference", ascending=False, inplace=True)
        df_diff.to_csv(os.path.join(output_dir, "pca_cluster_feature_differences.csv"), index=False)

        # 5. Plot
        plt.figure(figsize=(10, 0.5 * top_n))
        sns.heatmap(df_diff.head(top_n).set_index("Feature")[["Cluster 0 Mean", "Cluster 1 Mean"]],
                    annot=True, fmt=".2f", cmap="coolwarm", center=0)
        plt.title(f"Top {top_n} Feature Differences Between PCA Clusters")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"pca_cluster_feature_diff_top{top_n}.png"))
        plt.close()

        print(f"[INFO] Saved top {top_n} feature difference heatmap and CSV.")





        
        
    def get_cluster_data(self, cluster_id):
        """
        Returns rows from clustered_df corresponding to a specific PCA cluster.
        """
        if not hasattr(self, "clustered_df") or "pca_cluster" not in self.clustered_df.columns:
            raise ValueError("No clustered data found. Run explain_pca_clusters() first.")

        return self.clustered_df[self.clustered_df["pca_cluster"] == cluster_id].copy()


    
    def compare_clusters_on_feature(self, feature_index, output_dir="plots/compare_clusters"):
        """
        Compares the distribution of a given feature between PCA clusters 0 and 1.
        Saves a boxplot and prints statistical summary.
        
        Args:
        feature_index (int): Column index of the feature to compare.
        output_dir (str): Directory to save the resulting plot.
        """

        if self.clustered_df is None or "pca_cluster" not in self.clustered_df.columns:
            raise ValueError("Run explain_pca_clusters() before using this method.")

        os.makedirs(output_dir, exist_ok=True)

        feature_name = self.feature_labels.get(str(feature_index), f"Feature {feature_index}")

        df = self.clustered_df
        vals0 = df[df["pca_cluster"] == 0].iloc[:, feature_index].dropna()
        vals1 = df[df["pca_cluster"] == 1].iloc[:, feature_index].dropna()

        mean0 = vals0.mean()
        mean1 = vals1.mean()
        diff = abs(mean0 - mean1)

        t_stat, p_value = ttest_ind(vals0, vals1, equal_var=False)


        print("Feature: {feature_name} (index {feature_index})")
        print(f"Cluster 0 mean = {mean0:.3f}, Cluster 1 mean = {mean1:.3f}, |delta|= {diff:.3f}")
        print(f"t-statistic = {t_stat:.3f}, p-value = {p_value:.4f}")

        df_plot = pd.concat([
            df[df["pca_cluster"] == 0][[feature_index]].assign(Cluster="Cluster 0"),
            df[df["pca_cluster"] == 1][[feature_index]].assign(Cluster="Cluster 1")
        ])
        df_plot.columns = [feature_name, "Cluster"]

        # Plot
        plt.figure(figsize=(6, 4))
        sns.boxplot(data=df_plot, x="Cluster", y=feature_name)
        plt.title(f"{feature_name} ( {feature_index} ) by PCA Cluster")
        plt.tight_layout()

        # Save plot
        safe_name = feature_name.replace(" ", "_").replace("/", "_")
        filename = f"compare_feature_{feature_index}_{safe_name}.png"
        outpath = os.path.join(output_dir, filename)
        plt.savefig(outpath, dpi=150)
        plt.close()

        print(f"[INFO] Plot saved to: {outpath}")



        
    ######  not very useful at the moment
    def print_pca_feature_contributions(self, pca_model, feature_labels=None, top_n=10):
        top_n = int(top_n)  # Ensure it's an integer
        components = pca_model.components_
        for i, component in enumerate(components):
            print(f"\nTop {top_n} contributions to PC{i+1}:")
            indices = np.argsort(np.abs(component))[::-1][:top_n]
            for idx in indices:
                label = feature_labels.get(str(idx), f"Feature {idx}") if feature_labels else f"Feature {idx}"
                weight = component[idx]
                print(f"  {label}: {weight:.4f}")


                
    def plot_pca_feature_contributions(self, components, feature_names, top_n=10):
        """
        Plot the top contributing features for the first two principal components.
        
        Parameters:
        - components: PCA components_ array (n_components x n_features)
        - feature_names: dict of feature labels {str(index): label}
        - top_n: how many top features to show per component
        """

        n_components = min(2, components.shape[0])
        fig, axes = plt.subplots(1, n_components, figsize=(7 * n_components, 5), sharey=True)

        if n_components == 1:
            axes = [axes]

        for i in range(n_components):
            comp = components[i]
            indices = np.argsort(np.abs(comp))[::-1][:top_n]
            top_features = [feature_names.get(str(idx), f"Feature {idx}") for idx in indices]
            top_values = comp[indices]

            axes[i].barh(top_features, top_values, color='slateblue')
            axes[i].set_title(f'Top {top_n} contributors to PC{i + 1}', fontsize=12)
            axes[i].set_xlabel('Contribution')
            axes[i].invert_yaxis()
            axes[i].grid(True)

        plt.tight_layout()
        os.makedirs("plots", exist_ok=True)
        plt.savefig("plots/PCA_feature_contributions.png", dpi=150)
        plt.close()

                
    def run_factor_analysis(self, n_factors=5, max_features=30):
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



    def plot_feature_evolution(self, feature_col, output_dir="plots"):
        """
        Visualize evolution of a feature across Region and Dating.
        """
        os.makedirs(output_dir, exist_ok=True)        
        # Resolve feature name
        feature_name = self.feature_labels.get(str(feature_col), f"Feature {feature_col}")

        # Define columns
        region_col = 'Region' if 'Region' in self.df.columns else self.df.columns[1]
        date_col = 'Dating' if 'Dating' in self.df.columns else self.df.columns[3]

        df_temp = self.df[[region_col, date_col, feature_col]].dropna()

        # Ensure dating is numeric
        df_temp[date_col] = pd.to_numeric(df_temp[date_col], errors='coerce')
        df_temp = df_temp.dropna(subset=[date_col])

        # Compute average values per (region, date)
        grouped = df_temp.groupby([region_col, date_col])[feature_col].mean().reset_index()
#        grouped.columns = ["Region", "Date", "AverageFeature"]


        # Plot
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=grouped, x=date_col, y=feature_col, hue=region_col, marker="o",  palette="Set2")
        plt.title(f"Evolution of {feature_name} (column {feature_col}) by Region and Time")
        plt.xlabel("Dating (column 4)")
        plt.ylabel(f"Average presence of: {feature_name}")
        plt.legend(title="Region")
        plt.grid(True)
        plt.tight_layout()

        outpath = os.path.join(output_dir, f"evolution_feature_{feature_col}.png")
        plt.savefig(outpath, dpi=150)
        plt.close()

        print(f"Saved plot: {outpath}")
