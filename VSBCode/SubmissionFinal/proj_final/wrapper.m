%% Team : Honey, We Shrunk the Weights
%% This is a wrapper function to run the 4 models for proj_final

% trainX and test X = 35007 features, 
% However, for the demo below, 
% only 5000 word features are used to run the 4 models

function [yHatNB,yHatKNN,yHatSVM,yHatPCASVM] = wrapper(trainX, trainY, testX)
    trainXWords = trainX(:,1:5000);
    testXWords = testX(:,1:5000);

    yHatNB = nb(trainXWords, trainY, testXWords);
    yHatKNN = knn(trainXWords, trainY, testXWords);
    yHatSVM = svm(trainXWords, trainY, testXWords);
    yHatPCASVM = pcaSVM(trainXWords, trainY, testXWords);
end