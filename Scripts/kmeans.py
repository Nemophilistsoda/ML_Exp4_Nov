import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建保存图片的目录
output_dir = "ML_Exp4_Nov/pics/KMEANS"
os.makedirs(output_dir, exist_ok=True)

# 加载数据（本地csv，或用Kaggle链接下载）
df = pd.read_csv('ML_Exp4_Nov\Mall_Customers.csv')
df = df.drop('CustomerID', axis=1)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# 探索
print(df.describe())
sns.pairplot(df, hue='Gender')
plt.savefig(os.path.join(output_dir, 'pairplot.png'))  # 保存到指定路径
plt.close()

# 特征选择（年龄、收入、消费评分）
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 肘部法
inertias = []
silhouettes = []
K = range(2, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))

# 绘图
fig, ax1 = plt.subplots()
ax1.plot(K, inertias, 'bx-')
ax1.set_xlabel('K')
ax1.set_ylabel('Inertia', color='b')
ax2 = ax1.twinx()
ax2.plot(K, silhouettes, 'rx-')
ax2.set_ylabel('Silhouette Score', color='r')
plt.title('肘部法 + 轮廓系数确定最佳K')
plt.savefig(os.path.join(output_dir, 'elbow_silhouette.png'))  # 保存到指定路径
plt.close()

# 最佳K=5（实际运行结果）
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# 3D可视化
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], X['Age'],
                     c=clusters, cmap='viridis', s=50, edgecolor='k')
ax.set_xlabel('年收入 (k$)')
ax.set_ylabel('消费评分 (1-100)')
ax.set_zlabel('年龄')
plt.title('K-Means客户分群3D可视化')
plt.legend(*scatter.legend_elements(), title="簇")
plt.savefig(os.path.join(output_dir, 'kmeans_3d.png'))  # 保存到指定路径
plt.close()

# 簇中心反标准化查看真实含义
centers = scaler.inverse_transform(kmeans.cluster_centers_)
center_df = pd.DataFrame(centers, columns=X.columns)
print(center_df.round(1))