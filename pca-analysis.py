import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch


np.random.seed(42) # Set random seed
colors_df = ['#7fb3d5', '#154360', '#566573'] # color palette

# Read data
os.chdir('/Users/liokingv/Documents/Projects/PCA')
seeds_df = pd.read_csv("seeds_dataset.csv")

seeds_df.loc[seeds_df['seedType'].astype(str).str.contains('1', case=False), 'Variety'] = 'Kama'
seeds_df.loc[seeds_df['seedType'].astype(str).str.contains('2', case=False), 'Variety'] = 'Rosa'
seeds_df.loc[seeds_df['seedType'].astype(str).str.contains('3', case=False), 'Variety'] = 'Canadian'

# drop id column
seeds_no_id = seeds_df.drop(['ID', 'seedType', 'Variety'], axis=1)

# Standardize data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(seeds_no_id)

# Apply PCA
pca = PCA()
pca.fit(scaled_data)
pcs = pca.fit_transform(scaled_data)

# Look at explained variance ratio to see how much information each PC captures
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Plot explained variance
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Individual explained variance')
plt.step(range(1, len(explained_variance) + 1), cumulative_variance, where='mid', label='Cumulative explained variance')
plt.axhline(y=0.95, linestyle='--', color='r', label='95% threshold')
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by Principal Components')
plt.legend()
plt.tight_layout()
plt.show()

# Look at component loadings to see which original features contribute most to each PC
loadings = pca.components_.T
feature_names = seeds_no_id.columns

# Create a dataframe of the loadings
loadings_df = pd.DataFrame(
    loadings, 
    columns=[f'PC{i+1}' for i in range(loadings.shape[1])],
    index=feature_names
)

# Visualize the loadings with a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(loadings_df, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Feature Loadings for Principal Components')
plt.tight_layout()
plt.show()

# Calculate the absolute loadings and get the most important features for each PC
abs_loadings = abs(loadings_df)
# For each PC, find features with highest absolute loadings
for i in range(min(5, abs_loadings.shape[1])):  # Look at first 5 PCs or fewer
    pc = f'PC{i+1}'
    top_features = abs_loadings[pc].sort_values(ascending=False)
    print(f"\nTop features for {pc} (explains {explained_variance[i]:.2%} of variance):")
    print(top_features.head(3))  # Show top 3 features

# Calculate feature importance across all principal components
# This weights the loadings by the explained variance of each component
feature_importance = pd.DataFrame(
    data=np.zeros((len(feature_names), 1)),
    index=feature_names,
    columns=['Importance']
)

for i, exp_var in enumerate(explained_variance):
    feature_importance['Importance'] += abs_loadings[f'PC{i+1}'] * exp_var

# Sort features by overall importance
sorted_importance = feature_importance.sort_values('Importance', ascending=False)
print("\nOverall feature importance across all PCs:")
print(sorted_importance)

# Visualize overall feature importance
plt.figure(figsize=(10, 6))
sorted_importance.plot(kind='bar', figsize=(10, 6))
plt.title('Overall Feature Importance')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.tight_layout()
plt.show()

# Gap statistic function to estimate ideal number of clusters
def optimal_k_gap_statistic(data, k_max=10, B=10):

    def compute_gap_statistic(data, refs, k):
        km = KMeans(n_clusters=k, random_state=42).fit(data)
        disp = np.sum(np.min(cdist(data, km.cluster_centers_, 'euclidean'), axis=1))
        ref_disps = np.zeros(B)
        for i in range(B):
            random_reference = np.random.uniform(data.min(axis=0), data.max(axis=0), data.shape)
            km_ref = KMeans(n_clusters=k, random_state=42).fit(random_reference)
            ref_disp = np.sum(np.min(cdist(random_reference, km_ref.cluster_centers_, 'euclidean'), axis=1))
            ref_disps[i] = ref_disp
        gap = np.log(np.mean(ref_disps)) - np.log(disp)
        return gap

    gaps = []
    for k in range(1, k_max + 1):
        gap = compute_gap_statistic(data, None, k)
        gaps.append(gap)

    # Plot with labels
    plt.figure(figsize=(8, 5))
    ks = range(1, k_max + 1)
    plt.plot(ks, gaps, marker='o')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Gap Statistic')
    plt.title('Gap Statistic to Determine Optimal k')
    plt.grid(True)

    # Add labels on points
    for i, gap in enumerate(gaps):
        plt.text(ks[i], gap + 0.01, f"{gap:.2f}", ha='center', fontsize=9)

    plt.tight_layout()
    plt.show()

    optimal_k = np.argmax(gaps) + 1
    print(f"Optimal number of clusters: {optimal_k}")
    return optimal_k

# Use first 3 principal components for clustering
optimal_k_gap_statistic(pcs[:, :3], k_max = 10)

# Cluster analysis based on first three PCs
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(pcs[:, 0:3])

# Contingency table
# Add cluster labels to DataFrame
seeds_df['cluster'] = clusters
print("Contingency Table (Cluster vs Original Names):")
print(pd.crosstab(seeds_df['cluster'], seeds_df['Variety']))

# 2D plot of first two components
plt.figure(figsize=(8, 6))
sns.scatterplot(x=pcs[:, 0], y=pcs[:, 1], hue=clusters, palette=colors_df, s=100)
plt.title("2D Cluster Plot (PC1 vs PC2)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster")
plt.grid(True)
plt.show()

# 3D scatter plot of clusters
cluster_labels = np.unique(clusters)

# Assign colors based on cluster labels
cluster_colors = [colors_df[label] for label in clusters]

# 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(
    pcs[:, 0], pcs[:, 1], pcs[:, 2],
    c=cluster_colors, s=60, edgecolor='k'
)

ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')
ax.set_title("3D Cluster Plot (First Three PCs)")

# Custom legend
legend_elements = [Patch(facecolor=colors_df[i], edgecolor='k', label=f'Cluster {i+1}')
                   for i in range(len(colors_df))]
ax.legend(handles=legend_elements, title='Cluster', loc='upper right')

plt.tight_layout()
plt.show()