import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree  
from sklearn.neighbors import BallTree  

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
"""
NearestNeighbors用到的参数解释(非监督)
n_neighbors,默认值为5，表示查询k个最近邻的数目 
algorithm='auto',指定用于计算最近邻的算法，auto表示试图采用最适合的算法计算最近邻 
fit(X)表示用X来训练算法 
"""
nbrs = NearestNeighbors(n_neighbors=3, algorithm="ball_tree").fit(X)  
#返回距离每个点k个最近的点和距离指数，indices可以理解为表示点的下标，distances为距离  
distances, indices = nbrs.kneighbors(X)  
print(indices)
print(distances)

#输出的是求解n个最近邻点后的矩阵图，1表示是最近点，0表示不是最近点  
print (nbrs.kneighbors_graph(X).toarray())




#测试 KDTree  
'''
http://www.cnblogs.com/eyeszjwang/articles/2429382.html
leaf_size:切换到蛮力的点数。改变leaf_size不会影响查询结果， 
                          但能显著影响查询和存储所需的存储构造树的速度。 
                        需要存储树的规模约n_samples / leaf_size内存量。 
                        为指定的leaf_size，叶节点是保证满足leaf_size <= n_points < = 2 * leaf_size， 
                        除了在的情况下，n_samples < leaf_size。 
                         
metric:用于树的距离度量。默认'minkowski与P = 2（即欧氏度量）。 
                  看到一个可用的度量的距离度量类的文档。 
       kd_tree.valid_metrics列举这是有效的基础指标。 
'''
kdt = KDTree(X,leaf_size=30,metric="euclidean")  
print (kdt.query(X, k=3, return_distance=False))
  
  
#测试 BallTree  

bt = BallTree(X,leaf_size=30,metric="euclidean")  
print (bt.query(X, k=3, return_distance=False))

print('---------------------------------')

#有监督分类
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris  
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import svm

  
#查看iris数据集  
iris = load_iris()  
#print('iris',iris)
print ('iris.data',iris.data)
print ('iris.target',iris.target)
print (iris.data.shape, iris.target.shape)
print ('target_names',type(iris.target_names))
knn = KNeighborsClassifier().fit(iris.data, iris.target)
score = knn.score(iris.data, iris.target)
print('score',score)
predict = knn.predict([[0.1,0.2,0.3,0.4]])  
print ('predict',predict)
print(iris.target_names[predict])

X=iris.data
y=iris.target
X=preprocessing.scale(X)  #数据归一化
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
#X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.4, random_state=0)
'''
>>> X_train.shape, y_train.shape
((90, 4), (90,))
>>> X_test.shape, y_test.shape
((60, 4), (60,))
'''
knn = KNeighborsClassifier().fit(X_train, y_train)
score = knn.score(X_test, y_test)
print('score',score)

#支持向量机
#clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
#print('clf.score',clf.score(X_test, y_test))