%% Loading Dataset
% close all;
% clear all;
% load 'WordsData.mat';


%% Setting up data;
[iNumTrainRows, iNumTrainCols] = size(mWordsTrain);
[iNumTestRows, iNumTestCols] = size(mWordsTest);

% [~,mScore,~,~,vExplainedVar] = pca([mWordsTrain;mWordsTest]);
vCumSumExplainedVar = cumsum(vExplainedVar);
iMinPCs = min(find(vCumSumExplainedVar>=95));
mWordsTrainPCA = mScore(1:iNumTrainRows,1:iMinPCs);
mWordsTestPCA = mScore(iNumTrainRows+1:end,1:iMinPCs);

%% SVM Vanilla with PCA
% oSVMModel = fitcsvm(mWordsTrainPCA,vGendersTrain);
% vPredictedTrain = predict(oSVMModel,mWordsTrainPCA);
% iTrainError = sum(vPredictedTrain ~= vGendersTrain)/iNumTrainRows;

%% SVM with HeldOut
cvPartition = cvpartition(iNumTrainRows,'Holdout',.3);
trainIndices = training(cvPartition,1);
testIndices = test(cvPartition,1);

oSVMModels = fitcsvm(mWordsTrainPCA,vGendersTrain, 'CVPartition', cvPartition);
oSVMModel = oSVMModels.Trained{1};

vPredictedTrain = predict(oSVMModel,mWordsTrainPCA(trainIndices,:));
iTrainError = sum(vPredictedTrain ~= vGendersTrain(trainIndices))/length(find(trainIndices));

vPredictedTest= predict(oSVMModel,mWordsTrainPCA(testIndices,:));
iTestError = sum(vPredictedTest ~= vGendersTrain(testIndices))/length(find(testIndices));

%% SVM with KFold Validation

% cvPartition = cvpartition(200,'KFold',10);
% 
% cvSVMOutput = @(z)kfoldLoss(fitcsvm(cdata,grp,'CVPartition',cvPartition,...
%     'KernelFunction','rbf','BoxConstraint',exp(z(2)),...
%     'KernelScale',exp(z(1))));