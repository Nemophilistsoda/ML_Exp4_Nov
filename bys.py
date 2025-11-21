import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib as mpl
import warnings

warnings.filterwarnings('ignore')

# 设置字体和配色
mpl.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建保存图片的目录
output_dir = "ML_Exp4_Nov/pics/BYS"
os.makedirs(output_dir, exist_ok=True)

# 加载数据集：选取4个类别以简化
categories = ['alt.atheism', 'comp.graphics', 'rec.sport.baseball', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, remove=('headers', 'footers', 'quotes'))
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, remove=('headers', 'footers', 'quotes'))

X_train, y_train = newsgroups_train.data, newsgroups_train.target
X_test, y_test = newsgroups_test.data, newsgroups_test.target

print(f"训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")
print("类别分布:", pd.Series(y_train).value_counts().sort_index())

# 数据探索：词汇频率示例（前10词）
vectorizer = CountVectorizer(max_features=10, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
vocab = vectorizer.get_feature_names_out()
print("常见词汇:", vocab)

# 预处理：TF-IDF向量化
tfidf = TfidfVectorizer(max_features=5000, stop_words='english', min_df=2)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# 模型训练
models = {}
# 多项式朴素贝叶斯
models['Multinomial NB'] = MultinomialNB(alpha=1.0)
models['Multinomial NB'].fit(X_train_tfidf, y_train)

# 伯努利朴素贝叶斯
models['Bernoulli NB'] = BernoulliNB(alpha=1.0)
models['Bernoulli NB'].fit(X_train_tfidf, y_train)

# 超参数调优（多项式）
param_grid = {'alpha': [0.1, 1.0, 10.0], 'fit_prior': [True, False]}
grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='f1_macro')
grid_search.fit(X_train_tfidf, y_train)
models['Tuned Multinomial NB'] = grid_search.best_estimator_
print("最佳参数:", grid_search.best_params_)

# 基准：逻辑回归
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_tfidf, y_train)
models['Logistic Regression'] = lr

# 评估函数
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    try:
        y_prob = model.predict_proba(X_test)  # 多类预测概率
    except AttributeError:
        y_prob = None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"{model_name} 性能: 准确率={acc:.4f}, F1={f1:.4f}")

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=categories)
    disp.plot(cmap='coolwarm')  # 更好看的配色
    plt.title(f'{model_name} 混淆矩阵', fontsize=16)
    plt.savefig(os.path.join(output_dir, f'cm_{model_name.replace(" ", "_")}.png'), dpi=300)
    plt.show()

    if y_prob is not None:
        # 多类ROC需one-vs-rest，这里简化为单类示例
        fpr, tpr, _ = roc_curve((y_test == 0).astype(int), y_prob[:, 0])
        roc_auc = auc(fpr, tpr)
        print(f"AUC示例: {roc_auc:.4f}")
        RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc).plot()
        plt.title(f'{model_name} ROC曲线 (vs atheism)', fontsize=16)
        plt.savefig(os.path.join(output_dir, f'roc_{model_name.replace(" ", "_")}.png'), dpi=300)
        plt.show()

    return {'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1}

# 评估所有模型
results = {}
for name, model in models.items():
    results[name] = evaluate_model(model, X_test_tfidf, y_test, name)

# 可视化：特征重要性（用log概率示例）
feature_log_prob = models['Multinomial NB'].feature_log_prob_[0]  # atheism类
top_features = np.argsort(feature_log_prob)[-10:]
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_log_prob[top_features], y=[tfidf.get_feature_names_out()[i] for i in top_features], palette='coolwarm')
plt.title('Atheism类别Top 10特征词 (log P)', fontsize=16)
plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300)
plt.show()