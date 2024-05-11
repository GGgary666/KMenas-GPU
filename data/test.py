import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 随机生成一些数据
np.random.seed(0)  # 确保每次运行代码时生成的数据相同
X = np.random.rand(100, 2)  # 生成一个100x2的矩阵，表示100个二维数据点

# 创建KMeans实例，设置聚类数为3，并强制迭代50次
kmeans = KMeans(n_clusters=4, n_init=1, max_iter=50, init='random', verbose=1, tol=-1)

# 训练模型
kmeans.fit(X)

# 获取聚类中心
centroids = kmeans.cluster_centers_

# 获取每个数据点的标签
labels = kmeans.labels_

