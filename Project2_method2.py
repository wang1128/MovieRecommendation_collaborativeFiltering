__author__ = 'penghao'
from scipy import spatial
import numpy as np

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

def predictRate(simMatrix,dataMatrix,indexUsers):
    predictionList = []
    for i in range(0,1000):
        simIndex, sim, sumS = predict(simMatrix[i],10,dataMatrix[indexUsers])
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
    return predictionList

def testAcc(simMatrix,dataMatrix,pMatrix,indexofUser):
    predictionList = pMatrix[indexofUser]
    originalRate = dataMatrix[indexofUser]
    count = 0
    countAcc = 0
    for idx in range(0,1000):
        if originalRate[idx] != 0:
            count = count + 1
            if round(predictionList[idx]) == originalRate[idx]:
                countAcc = countAcc + 1
        else:
            continue
    return count, countAcc

def calPredictionMatrix():
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
    for i in y[groupNum]:
        count, countAcc =testAcc(simMatrix,dataMatrix,pMatrix,i)
        #print count, countAcc
        sumCount = sumCount + count
        sumCountAcc = sumCountAcc + countAcc
    return sumCountAcc/sumCount

dataMatrix= getDataMatrix()
aveList = getItemAve(dataMatrix)
#print aveList
#print len(aveList)
newM = getModifyMatrix(dataMatrix,aveList)
#print newM
#print newM.shape
r = calSim(newM[:,0],newM[:,1])
#print r

simMatrix = np.load('simMatrix.npy')
#simMatrix = SimMatrix(newM)


#pMatrix = np.array(calPredictionMatrix())
pMatrix = np.load('pMatrix.npy')
#np.save('pMatrix',pMatrix)
for i in range(0,10):
    print i
    acc = crossV(simMatrix,dataMatrix,pMatrix,i)
    print acc