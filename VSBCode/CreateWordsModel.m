%% Overall Plan
% NOTE: 
% Variable notations are rows, cols instead of X,Y - prevents confusion
% camelCase Variable Notation
% Hungarian Notation - prefix variables with following letters: 
% (i)nteger, (v)ector, (m)atrix, (o)bject 


% 1. Word counts need to be converted to relative frequencies:
% EachWordCountOfUser/TotalWordCountOfUser
% 2. Calculate PCAs (train + test data) and use enough PCs for 90-99% variance
% 3. Try different models with different features

%% Loading Dataset
close all;
clear all;
load 'WordsData.mat';


%% Setting up data; (1) and (2) above
[iNumTrainRows, iNumTrainCols] = size(mWordsTrain);
[iNumTestRows, iNumTestCols] = size(mWordsTest);

% Train - sum in rows, since each row depicts a user
vSumWordsTrain = sum(mWordsTrain,2);
mWordsTrainRelative = mWordsTrain./repmat(vSumWordsTrain,[1,iNumTrainCols]);

% Test - sum in rows, since each row depicts a user
vSumWordsTest = sum(mWordsTest,2);
mWordsTestRelative = mWordsTest./repmat(vSumWordsTest,[1,iNumTestCols]);

% Get PCA Scores and Variances 
% ** Un-comment below line before final submission **
%[~,mScore,~,~,vExplainedVar] = pca([mWordsTrainRelative;mWordsTestRelative]);
vCumSumExplainedVar = cumsum(vExplainedVar);
iMinPCs = min(find(vCumSumExplainedVar>=97));

% Build PCA dataset
mWordsTrainPCA = mScore(1:iNumTrainRows,1:iMinPCs);
mWordsTestPCA = mScore(iNumTrainRows+1:end,1:iMinPCs);

%% Naive Bayes Model With PCA & Cross-Validation
% oNBModelCV = fitcnb(mWordsTrainPCA,vGendersTrain, 'KFold', 10, 'Distribution','mn');
% iNBModelTestError = kfoldLoss(oNBModelCV);

%% DT Classification Model With PCA & Cross-Validation
% oDTModelCV = fitctree(mWordsTrainPCA,vGendersTrain, 'KFold', 10);
% iDTModelTestError = kfoldLoss(oDTModelCV);

%% KNN Classification Model With PCA & Cross-Validation
% oKNNModelCV = fitcknn(mWordsTrainPCA,vGendersTrain, 'KFold', 10);
% iKNNModelTestError = kfoldLoss(oKNNModelCV);

%% SVM Classification Model With PCA & Cross-Validation
% oSVMModelCV = fitcsvm(mWordsTrainPCA,vGendersTrain, 'KFold', 10, 'Standardize',true,'KernelFunction','RBF',...
%                       'KernelScale','auto');
% iSVMModelTestError = kfoldLoss(oSVMModelCV);

%% SVM Classification Model Without PCA and single Held-Out Dataset
oSVMModelCV = fitcsvm(mWordsTrainPCA,vGendersTrain, 'KernelFunction', 'gaussian', 'Holdout', 0.1);
vSVMModelTrainPredicted = kfoldPredict(oSVMModelCV);
iSVMModelTestError = kfoldLoss(oSVMModelCV);

%% Ensemble Classification Model With PCA & Cross-Validation
% oENSModelCV = fitensemble(mWordsTrainPCA,vGendersTrain, 'AdaBoostM1',1000,'Discriminant', ...
%                           'KFold', 10, 'LearnRate', 0.1);
% iENSModelTestError = kfoldLoss(oENSModelCV);

%% Predict the True Test Labels
% for iter = 1:10
%     mPredictedGenders(:,iter) = predict(oENSModelCV.Trained{iter}, mWordsTestPCA);
% end
%  vFinalPredictions = mode(mPredictedGenders,2);
% dlmwrite('submit.txt', vFinalPredictions);
