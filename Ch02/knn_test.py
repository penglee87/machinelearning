import matplotlib
import matplotlib.pyplot as plt
from numpy import *
import kNN
group,labels = kNN.createDataSet()
#print('group',group)
#print('labels',labels)
re = kNN.classify0([0,0],group,labels,3)  #4个参数分别为 输入向量、训练集、标签集、最近邻的个数，返回输入向量所属标签
print(re)


datingDataMat,datingLabels = kNN.file2matrix('datingTestSet2.txt')
#print(datingDataMat)
#print(datingLabels)
#print(datingDataMat.min(0))  #min() 返回所有值的最小值,min(0) 返回每列的最小值,min(1) 返回每行的最小值
kNN.datingClassTest()
fig = plt.figure()
ax = fig.add_subplot(111)
#ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
#ax.scatter(datingDataMat[:,0], datingDataMat[:,1],15.0*array(datingLabels), 15.0*array(datingLabels))  #参数分别为x轴值、y轴值、散点大小和散点颜色
ax.scatter(datingDataMat[:,0], datingDataMat[:,1],s=15.0*array(datingLabels), c=15.0*array(datingLabels))  #参数分别为x轴值、y轴值、散点大小和散点颜色
#plt.show()


def classifyPerson():
    resultList = ['not at all','in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print ("You will probably like this person: ",resultList[classifierResult - 1])