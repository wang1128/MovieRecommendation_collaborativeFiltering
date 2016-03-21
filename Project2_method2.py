__author__ = 'penghao'
from scipy import spatial
import numpy as np
import datetime

def getDataMatrix():
    infile = np.loadtxt('train.txt')
    dataMatrix = infile[:,:]
    return dataMatrix

def getItemAve(dataMatrix):
    aveList = []
    for idx in range(0,1000):
        new = np.copy(dataMatrix[:,idx]) # each column
        sum = 0.0
        count = 0.0
        for nums in new:
            if nums !=0:
                sum = sum + nums
                count= count + 1
        if count == 0:
            ave = 0.0

        else:
            ave = sum / count
        aveList.append(ave)
    return aveList

def getModifyMatrix(dataMatrix,aveList):# value = rating - average of rating
    newMatrix = np.copy(dataMatrix)
    for idx in range(0,1000):
        for i,element in enumerate(newMatrix[:,idx]):
            if element != 0:

                a = element - aveList[idx] #R - average of Rating
                newMatrix[:,idx][i] = a
            else:
                continue
    return newMatrix

def calSim(vetctor1, vector2):
    newtr =np.copy(vetctor1)
    newtest = np.copy(vector2)
    idxList = np.where(newtest == 0)
    idxList2 = np.where(newtest == 0.0)
    finallist = idxList2[0].tolist() + idxList[0].tolist()
    setList = set(finallist)
    final = list(setList)
    newtr[final] = 0
    newtest[final] = 0
    if np.count_nonzero(newtr) == 0:
        result = 0
    else:
        result = 1 - spatial.distance.cosine(newtr, newtest)
    return result

def SimMatrix(newM): # [0][:] means the first item similarity values with others
    simMatrix = np.zeros((1000, 1000))
    for idx in range(0,1000):
        for n in range(0,1000):
            if idx == n:
                sim = 0
            else:
                sim = calSim(newM[:,idx],newM[:,n])

            simMatrix[idx][n] = sim

    return simMatrix

def predict(simVector, numberOfSelect,dataVector): # the most largest value
    newSim = np.copy(simVector)
    for idx,element in enumerate(newSim):
        if dataVector[idx] == 0:
            newSim[idx] = 0
    simIndex = newSim.argsort()[-numberOfSelect:][::-1] # the largest n value index
    for idx, rate in enumerate(newSim):
        #sumSrate = 0
        sumS = 0
        similarityList = newSim[simIndex] #simlarity
        rateList = dataVector[simIndex]
        for idx,element in enumerate(similarityList):
            if rateList[idx]==0:
                sumS = sumS + 0
            else:
                sumS = sumS + abs(element)


    return simIndex, similarityList, sumS

def predictRate(simMatrix,dataMatrix,indexUsers): #predict rate
    predictionList = []
    for i in range(0,1000):
        simIndex, sim, sumS = predict(simMatrix[i],20,dataMatrix[indexUsers]) # After comparision 10 similarity item perform the best
        rateList = dataMatrix[indexUsers][simIndex]
        sumSR =0.0
        for idx, element in enumerate(rateList):
            sumSR = sumSR + element * sim[idx]

    #np.save('simMatrix.npy',simMatrix)
    #print simIndex
    #print sim
    #print sumS
    #print sumSR
        if sumS !=0:
            prediction = sumSR/ sumS
        else:
            prediction = 0
        predictionList.append(prediction)
        #print rateList
    #print predictionList
    return predictionList

def testAcc(simMatrix,dataMatrix,pMatrix,indexofUser,flag):
    predictionList = pMatrix[indexofUser]
    originalRate = dataMatrix[indexofUser]
    count = 0
    countAcc = 0
    if flag == 1:
        for idx in range(0,1000):
            if originalRate[idx] != 0:
                count = count + 1
                if round(predictionList[idx]) == originalRate[idx]: #or round(predictionList[idx]) == originalRate[idx] + 1 or round(predictionList[idx]) == originalRate[idx] - 1 :
                    #print predictionList[idx]
                    countAcc = countAcc + 1
            else:
                continue
    if flag == 2:
        for idx in range(0,1000):
            if originalRate[idx] != 0:
                count = count + 1
                if round(predictionList[idx]) == originalRate[idx] or round(predictionList[idx]) == originalRate[idx] + 1 or round(predictionList[idx]) == originalRate[idx] - 1 :
                    #print predictionList[idx]
                    countAcc = countAcc + 1
            else:
                continue
    return count, countAcc

def calPredictionMatrix(simMatrix,dataMatrix):
    pMatrix = []
    for i in range(0,200):
        print i
        p = predictRate(simMatrix,dataMatrix,i)
        pMatrix.append(p)
    return pMatrix

def crossV(simMatrix,dataMatrix,pMatrix,groupNum): # group num is 0-9

    x = np.arange(200)
    np.random.seed(0)
    np.random.shuffle(x) # x is a list of random number from 0 to 99
    y = np.split(x,10)
    sumCount = 0.0
    sumCountAcc =0.0
    accRateSum= 0.0
    for i in y[groupNum]:
        count, countAcc =testAcc(simMatrix,dataMatrix,pMatrix,i,2)

        #sumCount = sumCount + count
        #sumCountAcc = sumCountAcc + countAcc
        accRate = float(countAcc)/float(count)
        accRateSum +=accRate
    return accRateSum/float(len(y[groupNum]))#sumCountAcc/sumCount

def main():
    a = datetime.datetime.now().replace(microsecond=0)
    dataMatrix= getDataMatrix()
    aveList = getItemAve(dataMatrix)

    newM = getModifyMatrix(dataMatrix,aveList)

    r = calSim(newM[:,0],newM[:,1])

    simMatrix = np.load('simMatrix.npy')
    #simMatrix = SimMatrix(newM)
    #pMatrix = np.array(calPredictionMatrix(simMatrix,dataMatrix))
    pMatrix = np.load('pMatrix5.npy')

    #np.save('pMatrix20',pMatrix)
    for i in range(0,10):
        #print i
        acc = crossV(simMatrix,dataMatrix,pMatrix,i)
        print acc
    b = datetime.datetime.now()
    print(b-a)

if  __name__ =='__main__':
    main()
