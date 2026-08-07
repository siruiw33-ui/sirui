import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



def plot_decision_boundary(ax, clf, X, y, title):

    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))


    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    ax.contour(xx, yy, Z, colors='k', linewidths=1)


    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm,
                         edgecolors='k', s=50)


    if hasattr(clf, 'support_vectors_'):
        ax.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
                   s=100, facecolors='none', edgecolors='k')

    ax.set_xlabel('Sepal Length')
    ax.set_ylabel('Sepal Width')
    ax.set_title(title)


def plot_prediction_results_with_boundary(X_test, y_test, y_pred, clf, title):

    fig, ax = plt.subplots(figsize=(10, 6))

    # 绘制决策边界
    x_min, x_max = X_test[:, 0].min() - 0.5, X_test[:, 0].max() + 0.5
    y_min, y_max = X_test[:, 1].min() - 0.5, X_test[:, 1].max() + 0.5
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.2)
    ax.contour(xx, yy, Z, colors='k', linewidths=1)


    correct = (y_test == y_pred)
    colors = ['blue', 'green', 'red']


    for i in range(3):
        ax.scatter(X_test[y_test == i, 0], X_test[y_test == i, 1],
                   c='none', edgecolor=colors[i], marker='o',
                   s=70, label=f'Class {i} (True)', linewidths=1.5)


    for i in range(3):
        ax.scatter(X_test[(y_test == i) & correct, 0],
                   X_test[(y_test == i) & correct, 1],
                   c=colors[i], edgecolor='k', marker='o',
                   s=70, label=f'Class {i} (Correct)', linewidths=1)


    for i in range(3):
        for j in range(3):
            if i != j:
                mask = (y_test == i) & (y_pred == j)
                if np.any(mask):
                    ax.scatter(X_test[mask, 0], X_test[mask, 1],
                               c=colors[i], marker='x',
                               s=100, linewidths=1.5,
                               label=f'Class {i}→{j} (Wrong)')

    ax.set_xlabel('Sepal Length')
    ax.set_ylabel('Sepal Width')
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()



def main():

    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    labeled_ratio = 0.3
    n_labeled = int(labeled_ratio * len(y_train))
    X_labeled = X_train[:n_labeled]
    y_labeled = y_train[:n_labeled]
    X_unlabeled = X_train[n_labeled:]


    svm = SVC(kernel='linear', probability=True)


    svm.fit(X_labeled, y_labeled)

    threshold = 0.9
    while len(X_unlabeled) > 0:
        svm.fit(X_labeled, y_labeled)
        proba = svm.predict_proba(X_unlabeled)
        max_proba = np.max(proba, axis=1)
        confident_idx = np.where(max_proba >= threshold)[0]

        if len(confident_idx) == 0:
            break

        X_confident = X_unlabeled[confident_idx]
        y_confident = np.argmax(proba[confident_idx], axis=1)
        X_labeled = np.vstack((X_labeled, X_confident))
        y_labeled = np.hstack((y_labeled, y_confident))
        X_unlabeled = np.delete(X_unlabeled, confident_idx, axis=0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))


    initial_svm = SVC(kernel='linear').fit(X_train[:n_labeled], y_train[:n_labeled])
    plot_decision_boundary(ax1, initial_svm, X_train[:n_labeled], y_train[:n_labeled],
                           "Initial SVM (30% Labeled Data)")

    plot_decision_boundary(ax2, svm, X_labeled, y_labeled,
                           "Self-Trained SVM Decision Boundary")
    plt.tight_layout()
    plt.show()

    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Final Accuracy: {accuracy:.2f}")

    plot_prediction_results_with_boundary(X_test, y_test, y_pred, svm,
                                          f"Test Set Predictions (Accuracy: {accuracy:.2f})")


if __name__ == "__main__":
    main()