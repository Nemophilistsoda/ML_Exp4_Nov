import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, QuadraticDiscriminantAnalysis as QDA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import warnings

warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False  

# 创建保存图片的目录
output_dir = "ML_Exp4_Nov/pics/FISHER"
os.makedirs(output_dir, exist_ok=True)

# 加载数据
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = wine.target
print("数据集形状:", X.shape)
print("类别分布:\n", pd.Series(y).value_counts())

# 探索：相关性热图
plt.figure(figsize=(10, 8))
sns.heatmap(X.corr(), annot=False, cmap='coolwarm')
plt.title('特征相关性热图')
plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))  # 保存到指定路径
plt.show()

# 预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# 模型
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)

qda = QDA()
qda.fit(X_train_lda, y_train)

# 调优
param_grid = {'reg_param': [0, 0.1, 0.5]}
grid = GridSearchCV(QDA(), param_grid, cv=5)
grid.fit(X_train_lda, y_train)
best_qda = grid.best_estimator_
print("最佳参数:", grid.best_params_)

# 基准：SVM
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_lda, y_train)

# 评估
def evaluate(y_test, y_pred, name):
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"{name}: Acc={acc:.4f}, F1={f1:.4f}")
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=wine.target_names).plot(cmap='Blues')
    plt.title(f'{name} 混淆矩阵')
    plt.savefig(os.path.join(output_dir, f'cm_{name}.png'))  # 保存到指定路径
    plt.show()
    return acc, f1

y_pred_qda = best_qda.predict(X_test_lda)
acc_qda, f1_qda = evaluate(y_test, y_pred_qda, 'Tuned QDA')

y_pred_svm = svm.predict(X_test_lda)
acc_svm, f1_svm = evaluate(y_test, y_pred_svm, 'SVM')

# CV
cv_qda = cross_val_score(best_qda, X_train_lda, y_train, cv=5, scoring='f1_macro')
print(f"QDA CV F1: {cv_qda.mean():.4f} ± {cv_qda.std():.4f}")

# 可视化：决策边界
from sklearn.inspection import DecisionBoundaryDisplay
fig, ax = plt.subplots(figsize=(8, 6))
DecisionBoundaryDisplay.from_estimator(best_qda, X_train_lda, cmap='RdYlBu_r', ax=ax)
scatter = ax.scatter(X_train_lda[:, 0], X_train_lda[:, 1], c=y_train, edgecolors='k')
ax.set_xlabel('LD1')
ax.set_ylabel('LD2')
plt.title('LDA投影后决策边界')
handles, labels = scatter.legend_elements()
ax.legend(handles, wine.target_names)
plt.savefig(os.path.join(output_dir, 'decision_boundary.png'))  # 保存到指定路径
plt.show()