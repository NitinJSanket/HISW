function [vModels,vTrainAccuracy,vTestAccuracy] = ...
                createCVModels(mTrainData, vTrainLabels, iKFolds)
%% 
% INPUT
% Training Data, Testing Data and K-folds required

% OUTPUT - This function returns 3 vectors: 
% A vector of K-models, 
% A vector of K-corresponding training accuracies,
% A vector of K-corresponding test accuracies.

% You can call this function and then figure out how to vote for the best
% model using the available outputs

%%
[iNumSamples, iNumFeatures] = size(mTrainData);

oCVPartition = cvpartition(iNumSamples,'Holdout',iKFolds );
%oModels = fitcsvm(mTrainData,vTrainLabels, 'CVPartition', oCVPartition);
oModels = fitensemble(mTrainData,vTrainLabels,'AdaBoostM1',1000,'Tree','KFold', 2);

if iKFolds < 1
    iKFolds = 1;
end

vModels = oModels.Trained;
vTrainAccuracy = zeros(iKFolds,1);
vTestAccuracy = zeros(iKFolds,1)
for iter = 1:iKFolds
    trainIndices = training(oCVPartition,iter);
    testIndices = test(oCVPartition,iter);
    
    iNumTrainIndices = length(find(trainIndices));
    vPredictions = predict(vModels{iter},mTrainData(trainIndices,:));
    vTrainAccuracy(iter) = sum(vPredictions == vTrainLabels(trainIndices))/iNumTrainIndices;

    iNumTestIndices = length(find(testIndices));
    vPredictions = predict(vModels{iter},mTrainData(testIndices,:));
    vTestAccuracy(iter) = sum(vPredictions == vTrainLabels(testIndices))/iNumTestIndices;
end

end

