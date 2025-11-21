# 机器学习课程实验仓库
这个仓库包含了我的机器学习课程实验代码和相关可视化结果。主要使用 Mall_Customers.csv 数据集（一个常见的客户细分数据集），实现了几种经典的机器学习算法，包括贝叶斯分类器、Fisher 判别分析、K-Means 聚类和支持向量机（SVM）。每个实验都有对应的 Python 脚本和生成的图片（存放在 pics 文件夹下）。

## 仓库结构
- pics/ : 存放所有实验的可视化图片，按实验类型分文件夹。
  - BYS/ : 贝叶斯分类器的混淆矩阵、ROC 曲线、特征重要性图等。
  - FISHER/ : Fisher 判别分析的混淆矩阵、相关性热图、决策边界图等。
  - KMEANS/ : K-Means 聚类的肘部法图、轮廓图、3D 聚类图、配对图等。
  - SVM/ : SVM 的混淆矩阵、相关性热图、决策边界图等。
- Mall_Customers.csv : 实验数据集，包含客户年龄、收入、消费分数等特征。
- bys.py : 贝叶斯分类器实验（包括 Bernoulli NB、多项式 NB、逻辑回归等模型的实现和调优）。
- fisher.py : Fisher 判别分析实验（包括 SVM 和调优的 QDA 模型）。
- kmeans.py : K-Means 聚类实验（包括肘部法确定 K 值和可视化）。
- svm.py : 支持向量机实验（包括线性、多项式和 RBF 核的 SVM 模型及调优）。
## 实验描述
1. 贝叶斯分类器 (bys.py) 使用贝叶斯模型对客户数据进行分类。包括模型训练、评估和调优。生成图片包括混淆矩阵、ROC 曲线和特征重要性。
2. Fisher 判别分析 (fisher.py) 实现 Fisher 线性判别，结合 SVM 和 QDA 进行分类。生成相关性热图和决策边界可视化。
3. K-Means 聚类 (kmeans.py) 对客户数据进行无监督聚类，使用肘部法和轮廓分数确定最佳簇数。生成 3D 聚类图和配对散点图。
4. 支持向量机 (svm.py) 实现不同核函数的 SVM 分类器，包括线性、多项式和 RBF 核。生成混淆矩阵和决策边界图。
## 运行指南
### 环境要求
- Python 3.x
- 依赖库：numpy, pandas, scikit-learn, matplotlib, seaborn（可以通过 pip install -r requirements.txt 安装，如果有 requirements.txt 文件；否则手动安装）。
### 如何运行
1. 克隆仓库： git clone <repo-url>
2. 进入目录： cd ML_Exp4_Nov
3. 运行脚本，例如：
   - python bys.py （生成贝叶斯实验结果和图片）
   - python kmeans.py （生成 K-Means 结果和图片）
     每个脚本会自动处理数据、训练模型并保存图片到 pics 文件夹。
## 注意事项
- 数据集 Mall_Customers.csv 来自公开来源，用于客户细分分析。
- 所有图片都是脚本运行后自动生成的，用于结果可视化。
- 如果需要修改或扩展实验，可以直接编辑对应的 Python 脚本。
