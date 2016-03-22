#Movie_Recommendation_System
#Memory-based based on vector similarity
1. getDataMatrix() is to load the data to numpy array.<br />
2. setKValidation(num) is to set 10-fold cross-validation Matrix for training and testing.
3. calWeight(trainData, testVector,num) and calWeightList(k,trainData,testData) are used to
calculate the weight.
4. predict(k,w,trainData,testData) function is to predict the rate of each users.
5. testAccuracy(k,prediction,testData) function is to test the accuracy of prediction.
6. CrossValidation1(num) is to do the 10-fold cross validation.<br />

#Item-based correlation-based Similarity
1. getDataMatrix() is to load the data to numpy array.
2. getItemAve(dataMatrix) and getModifyMatrix(dataMatrix,aveList) are aim to generate new
Matrix
3. calSim(vetctor1, vector2) and SimMatrix(newM) are aim to generate similarity matrix of
different movies.
4. predict(simVector, numberOfSelect,dataVector), predictRate(simMatrix,dataMatrix,indexUsers)
and calPredictionMatrix(simMatrix,dataMatrix) are used for predicting the rate by using the
item base model.
5. testAcc(simMatrix,dataMatrix,pMatrix,indexofUser) and crossV(simMatrix,dataMatrix,pMatrix,groupNum)
are used for setting the 10-fold validation and testing the accuracy.<br />

# Method 3
1. The most functions are same as the memory based model. The way to calculate the weight is different from the method one.
2. In this method, the correlation-based Similarity is used instead of vector model.
3. calAve(vector) and calcorelation() are used for calculating the correlation.
4. setCorrelation(num) is used for generate the new corelation matrix.
