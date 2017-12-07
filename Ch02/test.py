from numpy import *
import operator
from os import listdir

def file2matrix(filename):
    fr = open(filename)
    print(fr)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    print(numberOfLines)
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return   
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        #print('line',line)
        line = line.strip()
        listFromLine = line.split('\t')
        #print(listFromLine)
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    #return returnMat,classLabelVector
    print(returnMat)
    
    
#datingDataMat,datingLabels = kNN.file2matrix('datingTestSet.txt')
file2matrix('datingTestSet2.txt')