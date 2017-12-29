__author__ = 'm.bashari'
import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

# 线性回归预测
def generate_data():
    np.random.seed(0)
    X, y = datasets.make_moons(200, noise=0.20)
    return X, y


def visualize(X, y, clf):
    # plt.scatter(X[:, 0], X[:, 1], s=40, c=y, cmap=plt.cm.Spectral)
    # plt.show()
    # clf.predict() 仅能接受单个向量，预测这个样本的类别
    plot_decision_boundary(lambda x: clf.predict(x), X, y)
    plt.title("Logistic Regression")


def plot_decision_boundary(pred_func, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    print(xx.shape)
    # Predict the function value for the whole gid
    # pred_func() 是 lambda x: clf.predict(x)
    # 近似于 z = lambda x: clf.predict(x), np.c_[xx.ravel(), yy.ravel()]
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)  #设置图的轮廓参数, 分别设置x轴、y轴、等高线和背景颜色
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
    plt.show()


def classify(X, y):
    clf = linear_model.LogisticRegressionCV()
    clf.fit(X, y)
    return clf


def main():
    X, y = generate_data()
    # print(X.shape)
    # visualize(X, y)
    clf = classify(X, y)
    visualize(X, y, clf)


if __name__ == "__main__":
    main()