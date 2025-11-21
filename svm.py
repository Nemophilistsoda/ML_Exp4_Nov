import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.decomposition import PCA
import matplotlib as plt

# 设置字体
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False  

# 创建保存图片的目录
output_dir = "ML_Exp4_Nov/pics/SVM"
os.makedirs(output_dir, exist_ok=True)

# 加载数据集
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # 0: malignant (恶性), 1: benign (良性)

# 可视化: 特征相关性热图 (选取前10维以简化)
plt.figure(figsize=(12, 8))
sns.heatmap(X.iloc[:, :10].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('前10个特征的相关性热图', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)  # 保存至指定路径
plt.show()

# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 模型训练
models = {}
models['Linear SVM'] = SVC(kernel='linear', C=1.0, random_state=42)
models['Linear SVM'].fit(X_train, y_train)

models['RBF SVM'] = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
models['RBF SVM'].fit(X_train, y_train)

models['Polynomial SVM'] = SVC(kernel='poly', C=1.0, degree=3, random_state=42)
models['Polynomial SVM'].fit(X_train, y_train)

# 超参数调优 (RBF)
param_grid = {'C': [0.1, 1, 10], 'gamma': [0.001, 0.01, 0.1]}
grid_search = GridSearchCV(SVC(kernel='rbf', random_state=42), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train, y_train)
models['Tuned RBF SVM'] = grid_search.best_estimator_

# 模型评估函数
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{model_name} 性能:")
    print(f"准确率: {acc:.4f}")
    print(f"精确率: {prec:.4f}")
    print(f"召回率: {rec:.4f}")
    print(f"F1分数: {f1:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['恶性', '良性'])
    disp.plot(cmap='Blues')
    plt.title(f'{model_name} 混淆矩阵', fontsize=16)
    plt.savefig(os.path.join(output_dir, f'cm_{model_name.replace(" ", "_")}.png'), dpi=300)
    plt.show()

    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        print(f"AUC: {roc_auc:.4f}")
        roc_disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        roc_disp.plot()
        plt.title(f'{model_name} ROC曲线', fontsize=16)
        plt.savefig(os.path.join(output_dir, f'roc_{model_name.replace(" ", "_")}.png'), dpi=300)
        plt.show()

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}

# 评估所有模型
results = {}
for name, model in models.items():
    results[name] = evaluate_model(model, X_test, y_test, name)

# 可视化决策边界 (PCA降维至2D)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

h = 0.02  # 网格步长
x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = models['Tuned RBF SVM'].predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')  # 更改为更好看的配色
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train, cmap='coolwarm', edgecolors='k')
plt.title('Tuned RBF SVM 决策边界 (PCA降维)', fontsize=16)
plt.xlabel('第一主成分', fontsize=14)
plt.ylabel('第二主成分', fontsize=14)
plt.savefig(os.path.join(output_dir, 'decision_boundary.png'), dpi=300)
plt.show()