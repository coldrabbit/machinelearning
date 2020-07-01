import numpy as np
from numpy import *
import csv
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    dataMat = []
    with open(fileName) as fileObject:
        reader = csv.DictReader(fileObject)
        listClass = []
        labelName = list(reader.fieldnames) 
        num = len(labelName) - 1

        for line in reader.reader:
            dataMat.append(line[:num])
            if line[-1] == 'male': #1代表male,0代表female
                gender = 1
            else:
                gender = 0
            listClass.append(gender)

        dataMat = np.array(dataMat).astype(float) 
        countVector = np.count_nonzero(dataMat, axis = 0)
        sumVector = np.sum(dataMat, axis=0) 
        meanVector = sumVector/countVector
        #求得每个特征的平均值

        for row in range(len(dataMat)):
            for col in range(num):
                if dataMat[row][col] == 0.0:
                    dataMat[row][col] = meanVector[col]
        #把缺失项用平均值填入

        dataSet = []
        for i in range(len(dataMat)):
            line = np.array(dataMat[i])
            dataSet.append(line)

        #构造训练集和测试集
        testSet = list(range(len(dataSet)))
        trainSet = []
        for i in range(2218):
            randomIndex = int(np.random.uniform(0, len(testSet)))
            trainSet.append(testSet[randomIndex])
            del testSet[randomIndex]
        #将数据集以7：3比例划分

        #训练集
        trainMat = []
        trainClasses = []
        for i in trainSet:
            trainMat.append(dataSet[i])
            trainClasses.append(listClass[i])

        #测试集
        testMat = []
        testClasses = []
        for i in testSet:
            testMat.append(dataSet[i])
            testClasses.append(listClass[i])

    return trainMat, trainClasses, testMat, testClasses, labelName

#贝叶斯分类器
def bayes(trainMatrix, listClasses):
    numTrainData = len(trainMatrix)
    numFeature = len(trainMatrix[0])

    p1class = sum(listClasses) / float(numTrainData)#训练集中男的比例

    trainData_1 = []
    key_1 = {}
    trainData_0 = []
    key_0 = {}

    #对训练集按男女进行分类
    for i in list(range(numTrainData)):
        if listClasses[i] == 1:
            trainData_1.append(trainMatrix[i])
        else:
            trainData_0.append(trainMatrix[i])

    trainData_1 = np.matrix(trainData_1)
    trainData_0 = np.matrix(trainData_0)

    #求男声参数
    for i in list(range(numFeature)):
        featureValues = np.array(trainData_1[:, i ]).flatten()
        featureValues = featureValues.tolist()
        key = {}
        key[mean] = np.mean(featureValues)
        key[std] = np.std(featureValues, ddof = 1)
        key[var] = np.var(featureValues)
        key_1[i] = key

    #求女生参数
    for i in list(range(numFeature)):
        featureValues = np.array(trainData_0[:, i ]).flatten()
        featureValues = featureValues.tolist()
        key = {}
        key[mean] = np.mean(featureValues)
        key[std] = np.std(featureValues, ddof = 1)
        key[var] = np.var(featureValues)
        key_0[i] = key
    
    return  key_1, key_0, p1class

#高斯分布方程
def equation(x, mean, std, var):
    k = 1/std
    N = -pow(x - mean, 2)/(2*var)
    return k*(np.e**N)

#预测测试集
def prediction(testVector, key_1, key_0, p1class):
    sum1 = 0
    sum0 = 0
    p0class = 1 - p1class
    for i in list(range(len(testVector))):
        sum1 = sum1 + np.log(equation(testVector[i], key_1[i][mean], key_1[i][std], key_1[i][var]))
        sum0 = sum0 + np.log(equation(testVector[i], key_0[i][mean], key_0[i][std], key_0[i][var]))
    p1 = sum1 + np.log(p1class)
    p0 = sum0 + np.log(p0class)
    if p1 > p0:
        return 1
    else: 
        return 0

#测试程序
def test():
    filename = 'D:/voice.csv'
    trainMat, trainClasses, testMat, testClasses, labelName = loadDataSet(filename)

    key_1, key_0, p1class = bayes(trainMat, trainClasses)

    count_1 = 0
    correctCount_1 = 0
    count_0 = 0
    correctCount_0 = 0

    for i in list(range(len(testMat))):
        testVector = testMat[i]
        result = prediction(testVector, key_1, key_0, p1class)
        if testClasses[i] == 1:
            count_1 = count_1 + 1
            if(result == 1):
                correctCount_1 = correctCount_1 + 1
        
        if testClasses[i] == 0:
            count_0 = count_0 + 1
            if(result == 0):
                correctCount_0 = correctCount_0 + 1

    return count_0, correctCount_0, count_1, correctCount_1

#绘制男女声正确率折线图
def picture(rate, number, title, Yname):
    x = range(1, number + 1)
    plt.plot(x, rate)
    plt.xlabel("Test times")
    plt.ylabel(Yname)
    plt.title(title)
    plt.show()

listNumber = 100 #测试100次求正确率平均值
rate0 = []
rate1 = []
for i in range(listNumber):
    tempc0, tempcc0, tempc1, tempcc1 = test()
    rate0.append(float(tempcc0/tempc0))
    rate1.append(float(tempcc1/tempc1))

print("Male voice correct rate:     ", mean(rate1))
print("Male voice error rate:     ", 1 - mean(rate1))
print("Female voice correct rate:   ", mean(rate0))
print("Female voice error rate:   ", 1 - mean(rate0))

picture(rate0, listNumber, "Female voice correct rate", "correct rate")
picture(rate1, listNumber, "Male voice correct rate", "correct rate")
      
