"""
K—Means 用于将数据集中的样本划分为K个不同的簇，
        核心思想：通过迭代，不断更新每个簇的中心点（质心）
        使每个样本到其所属的中心点距离之和最小
"""

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

np.random.seed(42)
X = np.vstack([
    np.random.normal([0,0],size=(50,2)),
    np.random.normal([5,5],size=(50,2)),
    np.random.normal([10,0],size=(50,2)),
])

kmeans = KMeans(n_clusters=3, random_state=42).fit(X)

labels = kmeans.labels_

centroids = kmeans.cluster_centers_

plt.scatter(X[:,0],X[:,1],c=labels,cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, c='red')

plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()