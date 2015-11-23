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

cvPartition = cvpartition(iNumSamples,'KFold',iKFolds );
oModels = fitcsvm(mTrainData,vTrainLabels, 'CVPartition', cvPartition);

vModels = oModels.Trained;
vTrainAccuracy = zeros(iKFolds,1);
vTestAccuracy = zeros(iKFolds,1);

for iter = 1:iKFolds
    trainIndices = training(cvPartition,iter);
    testIndices = test(cvPartition,iter);
    
    iNumTrainIndices = length(find(trainIndices));
    vPredictions = predict(vModels{iter},mTrainData(trainIndices,:));
    vTrainAccuracy(iter) = sum(vPredictions == vTrainLabels(trainIndices))/iNumTrainIndices;

    iNumTestIndices = length(find(testIndices));
    vPredictions = predict(vModels{iter},mTrainData(testIndices,:));
    vTestAccuracy(iter) = sum(vPredictions == vTrainLabels(testIndices))/iNumTestIndices;
end

end

