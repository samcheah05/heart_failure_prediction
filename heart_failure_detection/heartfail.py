import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Set style for presentation charts
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.size'] = 11

# ==============================================================================
# TASK 1: DATA SET PREPARATION & PREPROCESSING
# ==============================================================================
print("--- TASK 1: DATA EXPLORATION ---")

import os
import urllib.request

csv_file = "heart.csv"
if not os.path.exists(csv_file):
    print("Downloading heart.csv dataset...")
    url = "https://raw.githubusercontent.com/fedesoriano/data/main/heart.csv"
    urllib.request.urlretrieve(url, csv_file)
    print("Downloaded successfully!")

df = pd.read_csv(csv_file)

print(f"Total instances: {df.shape[0]}")
print(f"Total raw features: {df.shape[1] - 1}")
print("Target Distribution:")
print(df['HeartDisease'].value_counts(normalize=True))

# Separate features and target
X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Preprocessor: One-hot encode categoricals, standard scale numericals
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
    ]
)

# Processed matrix X_proc (needed for PCA, K-Means)
X_proc = preprocessor.fit_transform(X)
cat_encoded_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
all_feature_names = numerical_cols + list(cat_encoded_names)

# ==============================================================================
# TASK 2: EVALUATING SUPERVISED MODELS
# ==============================================================================
print("\n--- TASK 2: SUPERVISED MODEL EVALUATION ---")

# i. Train / Validation / Test split (60% Train, 20% Val, 20% Test)
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# Define Models
models = {
    "SVM (Linear)": Pipeline([
        ('prep', preprocessor),
        ('clf', SVC(kernel='linear', C=1.0, random_state=42))
    ]),
    "SVM (RBF)": Pipeline([
        ('prep', preprocessor),
        ('clf', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
    ]),
    "Random Forest": Pipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42))
    ])
}

# ii & iii: Train, Validate, and Evaluate on Test Set
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1
    })

results_df = pd.DataFrame(results)
print("\nTest Set Evaluation Metrics:")
print(results_df.to_string(index=False))

# iv: Diagnostic Learning Curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
train_sizes = np.linspace(0.1, 1.0, 8)

for ax, (name, model) in zip(axes, models.items()):
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X_train_val, y_train_val, 
        train_sizes=train_sizes, cv=5, scoring='f1', n_jobs=-1, random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    ax.plot(train_sizes_abs, train_mean, 'o-', color='#1f77b4', label='Training F1')
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#1f77b4')
    ax.plot(train_sizes_abs, val_mean, 's-', color='#ff7f0e', label='Cross-Val F1')
    ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#ff7f0e')

# ==============================================================================
# TASK 2 (PART IV): DIAGNOSTIC LEARNING CURVES
# ==============================================================================
print("\n--- GENERATING LEARNING CURVES ---")
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
train_sizes = np.linspace(0.1, 1.0, 8)

for ax, (name, model) in zip(axes, models.items()):
    train_sizes_abs, train_scores, val_scores = learning_curve(
        model, X_train_val, y_train_val, 
        train_sizes=train_sizes, cv=5, scoring='f1', n_jobs=-1, random_state=42
    )
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    ax.plot(train_sizes_abs, train_mean, 'o-', color='#1f77b4', label='Training F1')
    ax.fill_between(train_sizes_abs, train_mean - train_std, train_mean + train_std, alpha=0.15, color='#1f77b4')
    ax.plot(train_sizes_abs, val_mean, 's-', color='#ff7f0e', label='Cross-Val F1')
    ax.fill_between(train_sizes_abs, val_mean - val_std, val_mean + val_std, alpha=0.15, color='#ff7f0e')
    
    ax.set_title(f"Learning Curve: {name}", fontsize=13, fontweight='bold')
    ax.set_xlabel("Training Set Size", fontsize=11)
    ax.set_ylabel("F1 Score", fontsize=11)
    ax.set_ylim(0.70, 1.01)
    ax.legend(loc='lower right')

plt.tight_layout()
plt.savefig("slide_learning_curves.png", dpi=300)
plt.close()
print("Saved learning curves plot -> 'slide_learning_curves.png'")

# ==============================================================================
# TASK 3: EVALUATING UNSUPERVISED LEARNING (K-MEANS)
# ==============================================================================
print("\n--- TASK 3: K-MEANS CLUSTERING ---")
inertias = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X_proc)
    inertias.append(kmeans.inertia_)
    print(f"k={k} -> Distortion (Inertia): {kmeans.inertia_:.2f}")

plt.figure(figsize=(7, 5))
plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=7)
plt.axvline(x=2, color='r', linestyle='--', alpha=0.7, label='Optimal Elbow (k=2)')
plt.title("K-Means Elbow Method (Distortion vs. k)", fontsize=13, fontweight='bold')
plt.xlabel("Number of Clusters (k)", fontsize=11)
plt.ylabel("Inertia (Sum of Squared Distances)", fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig("slide_kmeans_elbow.png", dpi=300)
plt.close()
print("Saved elbow curve plot -> 'slide_kmeans_elbow.png'")

# Fit optimal k=2 and profile clusters
kmeans_final = KMeans(n_clusters=2, init='k-means++', n_init=20, random_state=42)
clusters = kmeans_final.fit_predict(X_proc)

df_analysis = df.copy()
df_analysis['Cluster'] = clusters
print("\nCluster Profiling (Mean values by cluster):")
cluster_summary = df_analysis.groupby('Cluster')[['Age', 'RestingBP', 'MaxHR', 'Oldpeak', 'HeartDisease']].mean()
print(cluster_summary)

# ==============================================================================
# TASK 4: DIMENSIONALITY REDUCTION (PCA)
# ==============================================================================
print("\n--- TASK 4: PRINCIPAL COMPONENT ANALYSIS ---")
pca = PCA()
pca.fit(X_proc)
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

for idx, var in enumerate(cumulative_variance, 1):
    print(f"PC 1 to {idx}: {var*100:.2f}% cumulative variance retained")

plt.figure(figsize=(7, 5))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'go-', linewidth=2)
plt.axhline(y=0.90, color='r', linestyle='--', label='90% Variance Retained')
plt.title("PCA: Cumulative Explained Variance", fontsize=13, fontweight='bold')
plt.xlabel("Number of Principal Components", fontsize=11)
plt.ylabel("Cumulative Variance Ratio", fontsize=11)
plt.legend()
plt.tight_layout()
plt.savefig("slide_pca_variance.png", dpi=300)
plt.close()
print("Saved PCA variance plot -> 'slide_pca_variance.png'")

# 2D PCA Projections
pca_2d = PCA(n_components=2)
X_pca_2d = pca_2d.fit_transform(X_proc)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
scatter1 = ax1.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=y, cmap='coolwarm', alpha=0.7, edgecolors='none')
ax1.set_title("True Ground Truth Labels (2D PCA)", fontsize=12, fontweight='bold')
ax1.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
ax1.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
fig.colorbar(scatter1, ax=ax1, label="Heart Disease (0=No, 1=Yes)")

scatter2 = ax2.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1], c=clusters, cmap='viridis', alpha=0.7, edgecolors='none')
ax2.set_title("K-Means Clusters (k=2 in 2D PCA)", fontsize=12, fontweight='bold')
ax2.set_xlabel(f"PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)")
ax2.set_ylabel(f"PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)")
fig.colorbar(scatter2, ax=ax2, label="Cluster")

plt.tight_layout()
plt.savefig("slide_pca_2d_projection.png", dpi=300)
plt.close()
print("Saved 2D PCA projection plot -> 'slide_pca_2d_projection.png'")