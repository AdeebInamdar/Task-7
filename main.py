import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# 1. Load and Preprocess Data
df = pd.read_csv("breast-cancer.csv")
df = df.drop(columns=['id'])  # Drop identifier column
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})  # Encode target: M=1, B=0

X = df.drop(columns=['diagnosis']).values
y = df['diagnosis'].values
feature_names = df.drop(columns=['diagnosis']).columns.tolist()

# 2. Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42, stratify=y)

# 4. Train, Tune, and CV (Linear Kernel)
param_grid_linear = {'C': [0.01, 0.1, 1, 10, 100]}
linear_svc = SVC(kernel='linear', random_state=42)
grid_linear = GridSearchCV(linear_svc, param_grid_linear, cv=5, scoring='accuracy', n_jobs=-1)
grid_linear.fit(X_train, y_train)

# 4. Train, Tune, and CV (RBF Kernel)
param_grid_rbf = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.01, 0.1, 1]}
rbf_svc = SVC(kernel='rbf', random_state=42)
grid_rbf = GridSearchCV(rbf_svc, param_grid_rbf, cv=5, scoring='accuracy', n_jobs=-1)
grid_rbf.fit(X_train, y_train)

# Final Test Set Evaluation
best_rbf_model = grid_rbf.best_estimator_
print(f"RBF Test set accuracy: {best_rbf_model.score(X_test, y_test):.4f}")

# 5. Visualize Decision Boundary (RBF)
X_2d = X_scaled[:, :2]  # Use the first two features (radius_mean and texture_mean)
viz_svc = SVC(kernel='rbf', C=best_rbf_model.C, gamma=best_rbf_model.gamma, random_state=42)
viz_svc.fit(X_2d, y)


def plot_decision_boundary(X, y, model, title, feature_names):
    markers = ('s', 'o')
    target_names = ['Benign', 'Malignant']
    colors = ('#0000FF', '#F08080')
    cmap = plt.cm.coolwarm

    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),
                           np.arange(x2_min, x2_max, 0.02))
    Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T).reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

    for cl in np.unique(y):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=colors[cl],
                    marker=markers[cl], label=target_names[cl],
                    edgecolor='black')

    plt.title(title)
    plt.xlabel(feature_names[0] + ' (scaled)')
    plt.ylabel(feature_names[1] + ' (scaled)')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig('svm_rbf_decision_boundary_2d_from_csv.png')
    plt.close()


plot_decision_boundary(X_2d, y, viz_svc,
                       f'RBF SVM Decision Boundary (C={viz_svc.C}, $\gamma$={viz_svc.gamma:.3f})',
                       feature_names)



