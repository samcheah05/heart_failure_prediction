# Heart Failure Risk Stratification: Machine Learning Evaluation

An end-to-end machine learning project evaluating supervised classifiers and unsupervised discovery algorithms for predicting heart disease and patient risk stratification.

---

## 1. Project Overview
This repository contains the experimental pipeline for evaluating:
- **Supervised Models:** Support Vector Machines (Linear vs. Gaussian/RBF kernels) benchmarked against Random Forest.
- **Diagnostics:** Learning curve bias-variance diagnostics on sample subsets.
- **Unsupervised Discovery:** K-Means clustering ($k$-selection via distortion elbow method).
- **Dimensionality Reduction:** Principal Component Analysis (PCA) for data compression and 2D visual projection.

---

## 2. Dataset Information
- **Source:** [Kaggle Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)
- **Instances:** 918 observations
- **Features:** 11 clinical features (demographics, resting vitals, exercise stress metrics, and electrocardiogram findings)
- **Target Variable:** `HeartDisease` (Binary: 0 = Normal, 1 = Heart Disease Present)
- **Class Distribution:** 55.34% Positive (508 cases) vs. 44.66% Negative (410 cases)

---

## 3. Key Experimental Findings

### Supervised Model Evaluation (Test Set, $n = 184$)

| Model | Accuracy | Precision | Recall (Sensitivity) | F1-Score |
|---|---|---|---|---|
| **SVM (Linear Kernel)** | **88.04%** | 87.04% | **92.16%** | **0.8952** |
| **SVM (Gaussian RBF)** | 87.50% | **87.62%** | 90.20% | 0.8889 |
| **Random Forest** | 86.96% | 86.11% | 91.18% | 0.8857 |

- **Linear Separability:** Linear SVM achieved the highest recall (**92.16%**) and F1-score (**0.8952**), showing that one-hot encoded categorical vitals create a clean, planar decision boundary.
- **Clinical Priority:** Recall and F1-score are prioritized over accuracy to minimize life-threatening false negatives.

### Unsupervised Discovery (K-Means Clustering)
- **Optimal Clusters:** Identified at $k = 2$ using the distortion/inertia elbow method.
- **Physiological Profiling:**
  - **Cluster 0 (Low Risk):** Mean age 48.2 years, mean max HR 153.0 bpm, low ST depression (0.32 mm) $\to$ **24.26% disease prevalence**.
  - **Cluster 1 (High Risk):** Mean age 58.4 years, mean max HR 121.8 bpm, high ST depression (1.41 mm) $\to$ **84.07% disease prevalence**.

### Dimensionality Reduction (PCA)
- **Variance Retention:** 8 principal components capture **90.49%** of cumulative variance, reducing dimensionality from 16 to 8 features (a 50% compression ratio).
- **2D Visual Insight:** Projections along PC1 and PC2 show class overlap near the center, highlighting the value of non-linear margin modeling.

---

## 4. Visual Diagnostics

| Metric | Output Visualization |
|---|---|
| **Learning Curves** | `slide_learning_curves.png` |
| **K-Means Elbow** | `slide_kmeans_elbow.png` |
| **PCA Variance Ratio** | `slide_pca_variance.png` |
| **2D Projections** | `slide_pca_2d_projection.png` |

---

## 5. Getting Started

### Prerequisites
- Python 3.10+
- Required packages:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
