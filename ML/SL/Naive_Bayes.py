"""
朴素贝叶斯：假设所有特征之间相互独立，但实际上会有影响，那么会有精度缺失
先验概率：根据以往经验得到的，不需要数据
后验概率：根据数据预测
联合概率：几个事件同时发生的概率（全概率公式）
朴素贝叶斯就是计算后验概率
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data[:, :2]  # 仅取前两个特征，方便可视化
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建并训练高斯朴素贝叶斯分类器
clf = GaussianNB()
clf.fit(X_train, y_train)

# 进行预测
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# 可视化预测结果
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, edgecolors='k', marker='o', s=80)
plt.title('Naive Bayes Classification of Iris Dataset')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()
