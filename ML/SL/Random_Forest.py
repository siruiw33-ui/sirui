"""
随机森林
核心思想：通过构建多个决策树，并将他们的预测结果结合起来，从而减少单个模型的过拟合，提高模型的泛化能力
基本步骤如下：
样本采样：通过Bootstrap抽样方法，从原始训练集中有放回地抽取多个子集，每个子集用于训练一个决策树
特征采样：在构建每个决策树时，对于每次分裂，只随机选择部分特征进行分裂选择，增加模型的多样性
决策树采样：对于每个子集，构建一颗决策树。决策树的深度通常较大，不进行剪枝。
结果融合：对于分类问题，采用多数投票法将所有树的预测结果进行投票
        对于回归问题，取所有树的预测平均值
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

rf_classifier = RandomForestClassifier(n_estimators = 100, random_state = 42)
rf_classifier.fit(x_train, y_train)

y_pred = rf_classifier.predict(x_test)

accuracy = accuracy_score(y_pred, y_test)
print(f"模型正确率: {accuracy}")

unique_labels = np.unique(y_pred)
true_count = [np.sum(y_test == label) for label in unique_labels]
pred_count = [np.sum(y_pred == label) for label in unique_labels]

x = np.arange(len(unique_labels))
width = 0.35

fig,ax = plt.subplots()
rects1 = ax.bar(x - width/2, true_count, width, label = 'Ture Labels', color = 'blue')
rects2 = ax.bar(x + width/2, pred_count, width, label = 'Predicted Labels', color = 'red')

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy = (rect.get_x() + rect.get_width() / 2, height),
                    xytext = (0, 3),
                    textcoords = 'offset points',
                    ha = 'center', va = 'bottom')

autolabel(rects1)
autolabel(rects2)

ax.set_ylabel('Count')
ax.set_title('True vs Predicted Labels by Class')
ax.set_xticks(x)
ax.set_xticklabels(iris.target_names)
ax.legend()

plt.show()

# 绘制 3x3 的决策树子图
fig, axes = plt.subplots(3, 3, figsize=(20, 20))
for i in range(3):
    for j in range(3):
        tree_index = i * 3 + j
        plot_tree(rf_classifier.estimators_[tree_index],
                  feature_names=iris.feature_names,
                  class_names=iris.target_names,
                  filled=True,
                  ax=axes[i, j])
        axes[i, j].set_title(f'Decision Tree {tree_index + 1}')

plt.tight_layout()
plt.show()



