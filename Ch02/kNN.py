'''
Created on Sep 16, 2010
kNN: k Nearest Neighbors

Input:      inX: vector to compare to existing dataset (1xN)
            dataSet: size m data set of known vectors (NxM)
            labels: data set labels (1xM vector)
            k: number of neighbors to use for comparison (should be an odd number)
            
Output:     the most popular class label

@author: pbharrin
'''
from numpy import *
import operator
from os import listdir

#4个参数分别为 输入向量、训练集、标签集、最近邻的个数，返回输入向量所属标签
def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  #将输入向量纵向扩展至训练集大小,然后减训练集
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)  #得到输入向量和每个训练样本的距离,axis=1表示按照横轴
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()  #距离排序
    classCount={}          
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  #py2
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #print('sortedClassCount[0][0]',type(sortedClassCount[0][0]))
    return sortedClassCount[0][0]

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

#将文本记录转化为numpy的解析程序
def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
    
#数据规一化,返回规一化后的数据集，特征度量向量，每个特征最小值组成的向量
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals
   
   
#按测试、训练(1/9)比例评估算法模型结果准确率
def datingClassTest():
    hoRatio = 0.10      #hold out 10%
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        #print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)
    

#将训练集作为测试集评估算法模型结果准确率
def datingClassTest2():
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i],normMat,datingLabels,5)
        #print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
    print(errorCount)

    
#对任一输入样本进行分类
def classifyPerson():
    resultList = ['not at all','in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print ("You will probably like this person: ",resultList[classifierResult - 1])
    
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)
    testFileList = listdir('testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr))
        if (classifierResult != classNumStr): errorCount += 1.0
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (errorCount/float(mTest)))
    
    
    
    
    
    
##############################################
import matplotlib
import matplotlib.pyplot as plt
group,labels = createDataSet()
#print('group',group)
#print('labels',labels)
re = classify0([0,0],group,labels,3)  #4个参数分别为 输入向量、训练集、标签集、最近邻的个数，返回输入向量所属标签
print('re',re)


datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
#print(datingDataMat)
#print(datingLabels)
#print(datingDataMat.min(0))  #min() 返回所有值的最小值,min(0) 返回每列的最小值,min(1) 返回每行的最小值
datingClassTest()
datingClassTest2()
fig = plt.figure()
ax = fig.add_subplot(111)  
#ax.scatter(datingDataMat[:,1], datingDataMat[:,2])
#ax.scatter(datingDataMat[:,0], datingDataMat[:,1],15.0*array(datingLabels), 15.0*array(datingLabels))  #作散点图，参数分别为x轴值、y轴值、散点大小和散点颜色
ax.scatter(datingDataMat[:,0], datingDataMat[:,1],s=15.0*array(datingLabels), c=15.0*array(datingLabels))  #作散点图，参数分别为x轴值、y轴值、散点大小和散点颜色
#plt.show()

#classifyPerson() #10,10000,0.5



print('---------------------------------')

#有监督分类
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris


datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
normMat, ranges, minVals = autoNorm(datingDataMat)
knn = KNeighborsClassifier().fit(normMat,datingLabels)
score = knn.score(normMat,datingLabels)
print('score',score)


print('---------------------------------')
#查看iris数据集  
iris = load_iris()  
#print('iris',iris)
#print('iris',iris.data, iris.target,iris.target_names,iris.DESCR,iris.feature_names)
knn = KNeighborsClassifier().fit(iris.data, iris.target)
score = knn.score(iris.data, iris.target)
print('score',score)
predict = knn.predict([[0.1,0.2,0.3,0.4]])  
print ('predict',predict)
print(iris.target_names[predict])
