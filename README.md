# Task 7: Support Vector Machines (SVM) for Breast Cancer Classification 🎗️

This repository contains the solution for Task 7 of the AI & ML Internship, focusing on implementing and tuning Support Vector Machines (SVM) for binary classification on the provided Breast Cancer dataset.

## 🎯 Objective

The goal of this task was to train, evaluate, and tune SVM classifiers using both **Linear** and **Radial Basis Function (RBF)** kernels to distinguish between Malignant (M) and Benign (B) cancer diagnoses.

## 🛠️ Tools and Libraries

* **Python**
* **Scikit-learn (sklearn):** For SVM models, scaling, and cross-validation.
* **NumPy:** For numerical operations.
* **Pandas:** For data loading and preprocessing.
* **Matplotlib:** For visualization of the decision boundary.

## 📂 Dataset

The analysis was performed using the provided `breast-cancer.csv` file, which contains 30 features (mean, standard error, and "worst" measurements of cell nucleus characteristics) and the `diagnosis` target column.

## ⚙️ Methodology

1.  **Data Preprocessing:** The `diagnosis` column was encoded, and the features were **standardized** using `StandardScaler`.
2.  **Data Splitting:** Data was split into training (70%) and testing (30%) sets.
3.  **Hyperparameter Tuning & Cross-Validation:**
    * `GridSearchCV` with **5-fold cross-validation (CV)** was used to find the optimal hyperparameters.
    * **Linear SVM:** Tuned $C$.
    * **RBF SVM:** Tuned $C$ and $\gamma$.
4.  **Evaluation:** Final accuracy scores were reported on the unseen test set.

## 📈 Key Results

The models achieved high performance, with the RBF kernel showing the best generalization accuracy.

| Kernel | Best Hyperparameters | Best CV Accuracy | Test Set Accuracy |
| :--- | :--- | :--- | :--- |
| **Linear** | $\mathbf{C=0.01}$ | $0.9698$ | $0.9532$ |
| **RBF** | $\mathbf{C=10}, \mathbf{\gamma=0.01}$ | $0.9697$ | $\mathbf{0.9766}$ |

## 🖼️ Decision Boundary Visualization (RBF Kernel)

The file `svm_rbf_decision_boundary_2d_from_csv.png` shows the non-linear separation boundary found by the RBF kernel using the `radius_mean` and `texture_mean` features.

## 🧠 Interview Questions (Theoretical Concepts)

| Concept | Explanation |
| :--- | :--- |
| **Support Vector** | Data points closest to the decision boundary; they define the hyperplane. |
| **C Parameter** | **Regularization term** controlling the trade-off between maximizing the margin and minimizing training errors. |
| **Kernels** | Functions (e.g., RBF) that map data to a high-dimensional space for linear separation via the "kernel trick." |
| **Overfitting** | Handled by **tuning $C$ and $\gamma$** via cross-validation to maintain a large margin and a smoother boundary. |
