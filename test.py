#coding:utf-8

# from python.Lib.packages.sklearn.tree import DecisionTreeClassifier
# from python.Lib.packages.matplotlib.pyplot import *
# from python.Lib.packages.sklearn.cross_validation import train_test_split
# from python.Lib.packages.sklearn.ensemble import RandomForestClassifier
# from python.Lib.packages.sklearn.externals.joblib import Parallel,delayed
# from python.Lib.packages.sklearn.tree import export_graphviz
# from python.Lib.packages.sklearn.datasets import load_iris
# import python.Lib.packages.pandas as pd


from sklearn.tree import DecisionTreeClassifier
from matplotlib.pyplot import *
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.joblib import Parallel,delayed
from sklearn.tree import export_graphviz
from sklearn.datasets import load_iris
import pandas as pd

def RandomForest(dir):
    # final = open('F:/test/final.dat' , 'r')
    data=pd.read_csv(dir)
    # data = [line.strip().split('\t') for line in final]
    feature=data[[i for i in range(8)]].values
    target=data[[8]].values
    # target1=[target[0][i] for i in range(len(target[0]))]
    # print feature
    # print target
    # feature = [[float(x) for x in row[3:]] for row in data]
    # target = [int(row[0]) for row in data]

    #拆分训练集和测试集
    # iris=load_iris()
    #
    # feature=iris.data
    # target=iris.target
    # print iris['target'].shape
    feature_train, feature_test, target_train, target_test = train_test_split(feature, target, test_size=0.1, random_state=42)

    #分类型决策树
    clf = RandomForestClassifier()

    #训练模型
    s = clf.fit(feature_train,target_train)
    print (s)

    #评估模型准确率
    r = clf.score(feature_test , target_test)
    print (r)

    print (u'判定结果：%s' % clf.predict(feature_test[0]))
    #print clf.predict_proba(feature_test[0])

    print (u'所有的树:%s' % clf.estimators_)

    print (clf.classes_)
    print (clf.n_classes_)

    print (u'各feature的重要性：%s' % clf.feature_importances_)
if __name__=="__main__":
    dir="Carseats.csv"
    RandomForest(dir)