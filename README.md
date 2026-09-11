# Breast Cancer Wisconsin: SVM vs K-Means

A machine learning analysis that compares **supervised classification** with Support Vector Machines (SVM) against **unsupervised clustering** with K-Means using the same Breast Cancer Wisconsin dataset.

## Project Overview

This project uses the **Wisconsin Diagnostic Breast Cancer (WDBC)** dataset to analyze two different machine learning approaches:

- **SVM:** predicts a known diagnosis using labeled observations.
- **K-Means:** searches for natural groups without using the diagnosis during clustering.

The goal is not only to compare model performance, but also to show the difference between predicting predefined classes and discovering structure directly from the data.

## Dataset

**Wisconsin Diagnostic Breast Cancer (WDBC)**  
Source: UCI Machine Learning Repository  
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

The dataset contains:

- **569 observations**
- **30 numerical predictor variables**
- One categorical target: `diagnosis`
- `B` = Benign
- `M` = Malignant

The predictors describe characteristics extracted from digitized images of breast mass samples, including radius, texture, perimeter, area, smoothness, concavity, and concave points.

## Analysis Questions

**Supervised learning:**  
Can the numerical characteristics of a sample be used to predict whether the tumor is benign or malignant?

**Unsupervised learning:**  
Do natural groups appear in the data when the diagnosis labels are completely ignored?

## Workflow

1. Load the original WDBC dataset from the ZIP file.
2. Inspect dimensions, data types, missing values, duplicates, and class distribution.
3. Remove the identifier and prepare the 30 numerical predictors.
4. Split the labeled data into 80% training and 20% testing using stratification.
5. Train SVM models with **Linear** and **RBF** kernels using `Pipeline` and `StandardScaler`.
6. Evaluate both models with Accuracy, Precision, Recall, F1-score, and a confusion matrix.
7. Compare different values of the SVM hyperparameter `C`.
8. Standardize the same predictors for K-Means without using the diagnosis.
9. Evaluate `k = 2` through `k = 6` using inertia and Silhouette Score.
10. Fit the final K-Means model.
11. Use PCA to project the 30-dimensional data into two dimensions.
12. Compare the discovered clusters with the real diagnoses only after clustering.

## SVM Results

| Model | Kernel | Accuracy | Precision | Recall | F1-score |
|---|---|---:|---:|---:|---:|
| Model 1 | Linear | 0.9649 | 1.0000 | 0.9048 | 0.9500 |
| Model 2 | RBF | **0.9737** | **1.0000** | **0.9286** | **0.9630** |

The **RBF SVM** achieved the best F1-score.

### Confusion Matrix

For the selected RBF model:

- 72 benign samples were correctly classified.
- 39 malignant samples were correctly classified.
- 0 benign samples were incorrectly classified as malignant.
- 3 malignant samples were incorrectly classified as benign.

This highlights why Accuracy alone is not enough: false negatives are especially relevant when a malignant observation is predicted as benign.

## SVM Hyperparameter Test

The RBF model was tested with different values of `C`:

| C | Accuracy | Precision | Recall | F1-score |
|---:|---:|---:|---:|---:|
| 0.1 | 0.9474 | 1.0000 | 0.8571 | 0.9231 |
| 1.0 | **0.9737** | **1.0000** | **0.9286** | **0.9630** |
| 10.0 | **0.9737** | **1.0000** | **0.9286** | **0.9630** |

`C = 1` provides the best observed performance without increasing model complexity unnecessarily.

## K-Means Results

K-Means was trained **without using `diagnosis`**.

The following values of `k` were evaluated:

| k | Silhouette Score |
|---:|---:|
| 2 | **0.3450** |
| 3 | 0.3144 |
| 4 | 0.2803 |
| 5 | 0.1585 |
| 6 | 0.1604 |

Based on the Silhouette Score and the inertia curve, **k = 2** was selected.

### Cluster Profiles

The clustering process produced two main profiles:

- **Cluster 0:** samples with larger average radius and area, as well as greater concavity and morphological irregularity.
- **Cluster 1:** samples with smaller average size and lower concavity and concave-point values.

These groups were discovered from geometric similarity only; diagnosis labels were not used by K-Means.

### Cluster vs. Real Diagnosis

After clustering was complete, the clusters were compared with the real labels:

| Cluster | Benign | Malignant |
|---|---:|---:|
| 0 | 7.41% | **92.59%** |
| 1 | **90.26%** | 9.74% |

There is a strong partial correspondence between the clusters and the real diagnoses, but they are not identical. K-Means minimizes geometric distance between observations and centroids rather than learning a known target class.

## PCA Visualization

Principal Component Analysis (PCA) was used to project the 30 scaled predictors into two dimensions.

- PC1 explained approximately **44.27%** of the variance.
- PC2 explained approximately **18.97%**.
- Together they represented approximately **63.24%** of the total variance.

The notebook includes PCA visualizations for both the K-Means clusters and the real diagnostic classes.

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab / Jupyter Notebook

## Repository Structure

```text
Breast-Cancer-Wisconsin-SVM-KMeans/
├── Breast_Cancer_Wisconsin.ipynb
├── breast+cancer+wisconsin+diagnostic.zip
├── README.md
├── requirements.txt
└── .gitignore
```

## Running the Project

### Google Colab

1. Open `Breast_Cancer_Wisconsin.ipynb` in Google Colab.
2. Select **Runtime → Run all**.
3. If the dataset ZIP is not already available in the Colab session, the notebook will request it automatically.
4. Upload `breast+cancer+wisconsin+diagnostic.zip`.
5. The notebook will extract and load `wdbc.data` automatically.

### Local Environment

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

Then open `Breast_Cancer_Wisconsin.ipynb` with Jupyter Notebook, JupyterLab, or VS Code.

## Main Conclusion

SVM is the appropriate approach when labeled examples are available and the objective is to **predict a known class**. K-Means is useful when labels are unavailable and the objective is to **explore natural structure or segmentation** in the data.

In this dataset, the supervised RBF SVM provided highly accurate diagnosis classification, while K-Means independently discovered two groups with a substantial, but not perfect, relationship to the real benign and malignant classes.
