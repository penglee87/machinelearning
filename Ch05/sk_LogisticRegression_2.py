from sklearn import datasets
from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import train_test_split  #弃用
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
#from PDC import plot_decision_regions
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

def versiontuple(v):
    return tuple(map(int, (v.split("."))))

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)  #绘制等高线图，对等高线间的区域进行填充
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        #print('idx',idx)
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.6, 
                    c=cmap(idx),
                    edgecolor='black',
                    marker=markers[idx], 
                    label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    edgecolor='black',
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')


############################
iris = datasets.load_iris()
x = iris.data[:,[2,3]]
y = iris.target  #一维数组,而不是一个1*n的二维数组
X_train,X_test,y_train,y_test = train_test_split(x , y, test_size=0.3, random_state = 0)
print(type(y_train))
print('y_train.shape',y_train.shape)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
print('X_train_std.shape',X_train_std.shape)
X_test_std = sc.transform(X_test)
Ir = LogisticRegression(C=100,random_state=0)
Ir.fit(X_train_std,y_train)
X_combined_std = np.vstack((X_train_std,X_test_std))
print('X_combined_std.shape',X_combined_std.shape)
y_combined = np.hstack((y_train,y_test))
print('y_combined.shape',y_combined.shape)

plot_decision_regions(X=X_combined_std,y=y_combined,classifier=Ir,test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.savefig('Iris.png')
plt.show()

print('X_test_std',X_test_std[0,:])
#a = Ir.predict_proba(X_test_std)
b = Ir.predict(X_test_std)
#print('a',a)
print('b',b)