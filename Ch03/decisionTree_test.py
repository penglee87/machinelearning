
import os
import graphviz 
from sklearn.externals.six import StringIO
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from numpy import *

iris = load_iris()
X=iris.data
y=iris.target
X=preprocessing.scale(X)  #数据标准化(结果均值为0,方差为1)
print(X.mean(axis=0))
print(X.std(axis=0))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
clf = tree.DecisionTreeClassifier(criterion = 'gini')  #仅能处理数值型特征
clf = clf.fit(X_train, y_train)

print(clf.score(X_test,y_test))  #模型准确率
print(clf.predict(X_test[:1,:]))  #预测样本对应分类
print(clf.predict_proba(X_test[:1, :]))  #预测划分到每个类的概率
print(clf.feature_importances_)  #特征重要性
print(clf.classes_)  #攻取所有标签类型


#dot_data = tree.export_graphviz(clf, out_file=None)
dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=iris.feature_names,  
                     class_names=iris.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)
#print('dot_data',type(dot_data),dot_data)
graph = graphviz.Source(dot_data)
#print('graph',type(graph),graph)
graph.render("iris")  #保存图源码并根据源码生成pdf文件存储图(生成pdf需要先安装Graphviz软件,并将其bin目录添加至环境变量)


'''
#group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
group = array([['a',1.1],['a',1.0],['b',0],['b',0.1]])
labels = ['A','A','B','B']
clf = tree.DecisionTreeClassifier(criterion = 'entropy')
clf = clf.fit(group, labels)
print(clf.score(group,labels))


fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels=['age','prescript','astigmatic','tearRate']
lensesTree = tree.DecisionTreeClassifier().fit(lenses[:][0:4],lenses[:][-1])

dot_data = tree.export_graphviz(lensesTree, out_file=None)
graph = graphviz.Source(dot_data)
graph.render("lenses")


with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)
os.unlink('iris.dot')



dot_data = StringIO() 
tree.export_graphviz(clf, out_file=dot_data) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
print('graph',type(graph),graph)
graph[0].write_pdf("iris.pdf") 


'''