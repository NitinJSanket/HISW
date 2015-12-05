Team: Honey, We Shrunk the Weights

In the folder:
* 4 Model files - nb.m, knn.m, svm.m, pcaSVM.m
* 1 Wrapper file - wrapper.m

The 4 model files have the same structure:
yHat = modelName(trainX, trainY, testX)

The wrapper file executes the above 4 models (on 5000 word features) and returns their respective predictions.  
It has the structure: 
[yHatNB,yHatKNN,yHatSVM,yHatPCASVM] = wrapper(trainX, trainY, testX)