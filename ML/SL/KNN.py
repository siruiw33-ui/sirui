import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np

iris = load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

y_pred = knn.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy}")

print("Detailed prediction results and true labels")
for i in range(len(y_test)):
    print(f"Sample {i+1}: True label = {y_test[i]}, Predicted label = {y_pred[i]}")

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(x_train[:, 0], x_train[:, 1], c=y_train, cmap='viridis', edgecolor='k', label='Training set')
plt.title('Training Set Distribution')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.legend()

plt.subplot(1, 2, 2)
correct_mask = y_pred == y_test
wrong_mask = y_pred != y_test

plt.scatter(x_test[correct_mask, 0], x_test[correct_mask, 1], c='g', s=100, label='Correct predictions')
plt.scatter(x_test[wrong_mask, 0], x_test[wrong_mask, 1], c='r', s=100, label='Wrong predictions')

plt.title('Test Set Prediction Results')
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.legend()

plt.tight_layout()
plt.show()