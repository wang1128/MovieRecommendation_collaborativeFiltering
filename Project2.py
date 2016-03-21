__author__ = 'penghao'
from scipy import spatial
import numpy as np
import datetime

def getDataMatrix():
    infile = np.loadtxt('train.txt')
    dataMatrix = infile[:,:]
    return dataMatrix

def setKValidation(num): #num is 0 - 9 10-fold validation
    dataMatrix = getDataMatrix()
    x = np.arange(200)
    np.random.seed(0)
    np.random.shuffle(x) # x is a list of random number from 0 to 199
    y = np.split(x,10) # y is a set of arrary that cut the list by 10 slides
    z = x.tolist()
    for element in y[num]:
        z.remove(element)

    trainData = dataMatrix[z]
    testData = dataMatrix[y[num]]
    #print trainData.shape, testData.shape
    return trainData, testData #180 for training, 20 for testing

def calWeight(trainData, testVector,num):
    '''
    newtr =np.copy(trainData)
    newtest = np.copy(testVector)
    wList = []
    idxList = np.where(newtest == 0)
    idxList2 = np.where(newtr[num] == 0.0)
    finallist = idxList2[0].tolist() + idxList[0].tolist()
    setList = set(finallist)
    final = list(setList)
    newtr[num][final] = 0
    newtest[final] = 0
    #print np.count_nonzero(newtr[num])
    #print np.count_nonzero(newtest)
    if np.count_nonzero(newtr[num]) == 0:
        result = 0
    else:
    '''
    result = 1 - spatial.distance.cosine(trainData[num], testVector)

    return result

def calWeightList(k,trainData,testData): # k is 0 -9
    wList = []
    for i in range(0,180):
        w= calWeight(trainData,testData[k],i)
        wList.append(w)
    return wList

def modifyTrainData(trainData):
    #print trainData[0]
    modifyTrain = np.copy(trainData)
    aveRate = []
    for idx in range(0,180):
        sum = 0.0
        count = 0.0
        for num in trainData[idx]:
            if num !=0:
                sum = sum + num
                count = count + 1
        ave = sum/count
        aveRate.append(ave)
    for idx in range(0,180):
        for i, num in enumerate(modifyTrain[idx]):
            if num !=0:
                modifyTrain[idx][i] = num - aveRate[idx]

    #print trainData[0]
    #print modifyTrain[0]
    #print aveRate
    return modifyTrain

def predict(k,w,trainData,testData): #testdata 5
    sum = 0.0
    count = 0.0
    for element in testData[k]:
        if element != 0:
            sum = sum + element
            count = count + 1
    testave = sum/ count

    mTrain = modifyTrainData(trainData)
    predictList = []
    sumW = 0
    for num in w:
        sumW = sumW + abs(num)
    #print sumW
    sumAbove = []
    rateList = []
    for i in range(0,1000):
        for idx in range(0,180):
            sumAll = 0
            rate = mTrain[:,i]
            weight = w[idx]
            for num in rate:
                sumAll = sumAll + num*weight
        predictRate = testave + sumAll/sumW
        rateList.append(predictRate)
        sumAbove.append(sumAll)

    #print sumAbove
    #print rateList
    return rateList

def testAccuracy(k,prediction,testData):
    count = 0
    countAcc = 0
    for idx, num in enumerate(testData[k]):
        if num !=0:
            count += 1
            #print round(predict[idx])
            if round(prediction[idx]) == num or round(prediction[idx]) == num + 1 or round(prediction[idx]) == num - 1:

                countAcc += 1

    #print count
    #print countAcc
    return count, countAcc

def CrossValidation1(num):
    trainData, testData = setKValidation(num)
    totalRateNum = 0.0
    accPrediction = 0.0
    for i in range(0,10):
        #i=8
        w = calWeightList(i,trainData,testData)

        preList = predict(i,w,trainData,testData)
        total, acc = testAccuracy(i,preList,testData)
        totalRateNum += total
        accPrediction += acc
    print accPrediction/totalRateNum

def main():
    a = datetime.datetime.now().replace(microsecond=0)
    for i in range(10):
        #print i
        CrossValidation1(i)
    b = datetime.datetime.now()
    print(b-a)


if  __name__ =='__main__':
    main()