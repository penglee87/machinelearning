'''
Created on Oct 12, 2010
Decision Tree Source Code for Machine Learning in Action Ch. 3
@author: Peter Harrington
'''
from math import log
import operator

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['nosurfacing','flippers']
    #change to discrete values
    return dataSet, labels

#计算数据集香农熵,熵越小,纯度越高(平均分布时,纯度最低)
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet: #the the number of unique elements and their occurance
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2) #log base 2
    return shannonEnt
    
#按照给定特征划分数据集,返回此特征对应值的除去此特征列后的所有样本
def splitDataSet(dataSet, axis, value):  #待划分数据集、使用的特征、特征的返回值
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
    
#选择最好的划分数据集的特征列
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #get a set of unique values
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)     
        #print('i,value,newEntropy',i,value,newEntropy)
        infoGain = baseEntropy - newEntropy     #calculate the info gain; ie reduction in entropy
        if (infoGain > bestInfoGain):       #compare this to the best gain so far
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

#返回标签值最多的标签
def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    #print('classCount',classCount)
    #sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)  #py2
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)  #py3
    #print('sortedClassCount',sortedClassCount)
    return sortedClassCount[0][0]

#创建决策树,labels用于存储列名
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0]#stop splitting when all of the classes are equal
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            
    
#使用决策树对输入样本进行分类,入参分别为之前训练生成的tree,列标签,输入的样本
def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    print('firstStr',type(firstStr),firstStr)
    print('featLabels',type(featLabels),featLabels)
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb+')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    fr = open(filename,'rb+')
    return pickle.load(fr)
    



##############################
if __name__ == "__main__":
    myDat,labels=createDataSet()
    print('myDat',myDat)
    print('labels',labels)
    print('calcShannonEnt',calcShannonEnt(myDat))
    print('splitDataSet',splitDataSet(myDat, 1, 1))
    print('chooseBestFeatureToSplit',chooseBestFeatureToSplit(myDat))
    
    #classList = [example[-1] for example in myDat]
    #majorityCnt(classList)
    
    myDat,labels=createDataSet()
    print('labels',labels)
    myTree = createTree(myDat,labels)  #myDat,labels值会被更改,所以后面测试需重新赋值
    print('myTree',myTree)
    myDat,labels=createDataSet()
    print('labels',labels)
    print('classify',classify(myTree,labels,[1,0]))
    
    
    storeTree(myTree,'classifierStorage.txt')
    loadtree=grabTree('classifierStorage.txt')
    print('classify',classify(loadtree,labels,[1,0]))
    
    myDat[0][-1]='maybe'
    print(myDat)
    print(calcShannonEnt(myDat))
    print(chooseBestFeatureToSplit(myDat))
    
    
    
    
    
    fr=open('lenses.txt')
    lenses=[inst.strip().split('\t') for inst in fr.readlines()]
    lensesLabels=['age', 'prescript', 'astigmatic', 'tearRate']
    lensesTree = createTree(lenses,lensesLabels)
    print('lensesTree',lensesTree)