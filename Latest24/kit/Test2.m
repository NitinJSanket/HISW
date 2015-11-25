clc
clear all
close all

addpath(genpath('./RF'));
addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');
addpath('./liblinear');

addpath('../../GIST/');
addpath('../MIToolbox/');
addpath('../FEAST/');

load('GISTFeatures.mat');
load('GISTFeaturesT.mat');

% addpath('./drtoolbox/', './drtoolbox/techniques/');
% rmpath('./drtoolbox/', './drtoolbox/techniques/');
%% Read Everything
TrainImgs = csvread([pwd, '/train/images_train.txt']);
TrainLabels = csvread([pwd, '/train/genders_train.txt']);
TrainWords = csvread([pwd, '/train/words_train.txt']);
TrainImgFeatures = csvread([pwd, '/train/image_features_train.txt']);

TestImgs = csvread([pwd, '/test/images_test.txt']);
TestWords = csvread([pwd, '/test/words_test.txt']);
TestImgFeatures = csvread([pwd, '/test/image_features_test.txt']);

disp('Reading Data Complete....');
%% Split into CV Training and Testing Sets
NTest = 1000;
NTrain = length(TrainLabels)-NTest;
CVTrainImgs = TrainImgs(1:NTrain,:);
CVTestImgs = TrainImgs(NTrain+1:end, :);


CVTrainLabelsOrg = TrainLabels(1:NTrain,:);
CVTestLabelsOrg = TrainLabels(NTrain+1:end, :);

CVTrainWords = TrainWords(1:NTrain,:);
CVTestWords = TrainWords(NTrain+1:end, :);

CVTrainImgFeatures = TrainImgFeatures(1:NTrain,:);
CVTestImgFeatures = TrainImgFeatures(NTrain+1:end, :);

%% PCA
PCAFlag = 0;
NC = 25;
CVTrainWords = bsxfun(@rdivide, bsxfun(@minus, CVTrainWords, mean(CVTrainWords)), var(CVTrainWords) + 1e-10);
CVTestWords = bsxfun(@rdivide, bsxfun(@minus, CVTestWords, mean(CVTestWords)), var(CVTestWords) + 1e-10);
if(PCAFlag)
    [coeffTrain,scoreTrain,latentTrain] = pca(TrainWords);
    scoreCVTrain = scoreTrain(1:NTrain, 1:NC);
    scoreCVTest = scoreTrain(NTrain+1:end, 1:NC);
    figure,
    plot(cumsum(latentTrain)./sum(latentTrain));
    
    disp('PCA Complete....');
else
    scoreCVTrain = CVTrainWords;
    scoreCVTest = CVTestWords;
end

%% GIST
param.imageSize = [100 100]; % it works also with non-square images
param.orientationsPerScale = [8 8 8 8];
param.numberBlocks = 4;
param.fc_prefilt = 4;

% Computing gist requires 1) prefilter image, 2) filter image and collect
% output energies
GISTFeaturesT = [];
for i = 1:size(TestImgs,1)
    disp(i);
  cur_row=TestImgs(i,:);
  cur_img=reshape(cur_row,[100 100 3]);
[gist1, param] = LMgist(cur_img, '', param);
GISTFeaturesT = [GISTFeaturesT; gist1];
end

%% DCT
% scoreCVTrain = CVTrainWords;
% scoreCVTest = CVTestWords;
% for i = 1:size(scoreCVTrain,1)
%     scoreCVTrain(i,:) = dct(CVTrainWords(i,:));
% end
%
% for i = 1:size(scoreCVTest,1)
%     scoreCVTest(i,:) = dct(CVTestWords(i,:));
% end
%
% scoreCVTrain = sort(abs(scoreCVTrain),2);
% scoreCVTest = sort(abs(scoreCVTest),2);
% scoreCVTrain = scoreCVTrain(:,1:1000);
% scoreCVTest = scoreCVTest(:,1:1000);

%% Feature Selection
% scoreCVTrain = CVTrainWords;
% scoreCVTest = CVTestWords;
% selectedIndices = feast('mrmr',150,scoreCVTrain,CVTrainLabels); %% selecting the top 5 features using the jmi algorithm
% scoreCVTrain1 = CVTrainWords(:,selectedIndices);
% scoreCVTest1 = CVTestWords(:,selectedIndices);
% 
% disp('Feature Selection Complete....');

LowerLimAge = [0,10,16,19,30,40];
UpperLimAge = [9,15,18,29,39,80];

FinalPredM1 = [];
FinalPredM2 = [];
FinalPredM12 = [];

NBoost1 = [40, 25, 30, 800,  500, 100];
NBoost2 = [40, 25, 30, 300,  50, 100];
SFFlag1 = [1,   1,  1,   0,   0,  1];
SFNos1 = [20, 150, 50, 400, 100, 100];
GISTFlag = [1,1,1,1,0,1];

Sum = 0;
%%
for AgeGroup = 1:6
    disp(['Building Model for Age group between ', num2str(LowerLimAge(AgeGroup)), ' to ', num2str(UpperLimAge(AgeGroup))]);
    TrainIdxs = TrainImgFeatures(:,1)>=LowerLimAge(AgeGroup) & TrainImgFeatures(:,1)<=UpperLimAge(AgeGroup);
    TestIdxs = TestImgFeatures(:,1)>=LowerLimAge(AgeGroup) & TestImgFeatures(:,1)<=UpperLimAge(AgeGroup);
    Sum = Sum+sum(TestIdxs);
    disp(Sum);
    
    TrainLabels = TrainLabelsOrg(TrainIdxs,:);
%% Words
scoreTrain1 = TrainWords(TrainIdxs,:);
scoreTest1 = TestWords(TestIdxs,:);
%% Training

if(SFFlag1(AgeGroup)==1)
selectedIndices = feast('mrmr',SFNos1(AgeGroup),scoreTrain1,TrainLabels); %% selecting the top 5 features using the jmi algorithm
scoreTrain1 = scoreTrain1(:,selectedIndices);
scoreTest1 = scoreTest1(:,selectedIndices);
end

Ensemble1{AgeGroup} = fitensemble(scoreTrain1,TrainLabels,'AdaBoostM1',NBoost1(AgeGroup),'Tree');
[TrainPred1, TrainScore1] = predict(Ensemble1{AgeGroup}, scoreTrain1);

disp('Training Complete....');
%% Testing

[TestPred1, TestScore1] = predict(Ensemble1{AgeGroup},scoreTest1);
disp('Testing Complete....');

FinalPredM1 = [FinalPredM1;TestPred1];
%% Use GIST Features
scoreTrain2 = GISTFeatures;
scoreTest2 = GISTFeaturesT;
scoreTrain2 = scoreTrain2(TrainIdxs,:);
scoreTest2 = scoreTest2(TestIdxs,:);

%% Training

Ensemble2{AgeGroup} = fitensemble(scoreTrain2,TrainLabels,'AdaBoostM1',NBoost2(AgeGroup),'Tree');
[TrainPred2, TrainScore2] = predict(Ensemble2{AgeGroup}, scoreTrain2);

disp('Training Complete....');
%% Testing

[TestPred2, TestScore2] = predict(Ensemble2{AgeGroup},scoreTest2);
disp('Testing Complete....');

FinalPredM2 = [FinalPredM2;TestPred2];
%% Combine the 2 results
TestScore1R = TestScore1(:,1)./max(abs(TestScore1(:,1)));
TestScore2R = TestScore2(:,1)./max(abs(TestScore2(:,1)));
TestScore12R = (TestScore1R + TestScore2R.*GISTFlag(AgeGroup)) < 0;

FinalPredM12 = [FinalPredM12;TestScore12R];
%% Display Accuracy

disp(['Train Accuracy Words ', num2str(sum(TrainPred1==TrainLabels)/sum(TrainIdxs))]);
disp(['Train Accuracy GIST ', num2str(sum(TrainPred2==TrainLabels)/sum(TrainIdxs))]);

end


%%
% for AgeGroup = 1:6
%     disp(['Building Model for Age group between ', num2str(LowerLimAge(AgeGroup)), ' to ', num2str(UpperLimAge(AgeGroup))]);
%     TrainIdxs = CVTrainImgFeatures(:,1)>=LowerLimAge(AgeGroup) & CVTrainImgFeatures(:,1)<UpperLimAge(AgeGroup);
%     TestIdxs = CVTestImgFeatures(:,1)>=LowerLimAge(AgeGroup) & CVTestImgFeatures(:,1)<UpperLimAge(AgeGroup);
%     
%     CVTrainLabels = CVTrainLabelsOrg(TrainIdxs,:);
%     CVTestLabels = CVTestLabelsOrg(TestIdxs,:);
% %% Words
% scoreCVTrain1 = CVTrainWords(TrainIdxs,:);
% scoreCVTest1 = CVTestWords(TestIdxs,:);
% %% CV Training
% 
% if(SFFlag1(AgeGroup)==1)
% selectedIndices = feast('mrmr',SFNos1(AgeGroup),scoreCVTrain1,CVTrainLabels); %% selecting the top 5 features using the jmi algorithm
% scoreCVTrain1 = scoreCVTrain1(:,selectedIndices);
% scoreCVTest1 = scoreCVTest1(:,selectedIndices);
% end
% 
% Ensemble1{AgeGroup} = fitensemble(scoreCVTrain1,CVTrainLabels,'AdaBoostM1',NBoost1(AgeGroup),'Tree');
% [CVTrainPred1, CVTrainScore1] = predict(Ensemble1{AgeGroup}, scoreCVTrain1);
% 
% disp('CV Training Complete....');
% %% CV Testing
% 
% [CVTestPred1, CVTestScore1] = predict(Ensemble1{AgeGroup},scoreCVTest1);
% disp('CV Testing Complete....');
% 
% FinalPredM1 = [FinalPredM1;CVTestPred1];
% %% Use GIST Features
% scoreCVTrain2 = GISTFeatures(1:NTrain,:);
% scoreCVTest2 = GISTFeatures(NTrain+1:end,:);
% scoreCVTrain2 = scoreCVTrain2(TrainIdxs,:);
% scoreCVTest2 = scoreCVTest2(TestIdxs,:);
% 
% %% CV Training
% 
% Ensemble2{AgeGroup} = fitensemble(scoreCVTrain2,CVTrainLabels,'AdaBoostM1',NBoost2(AgeGroup),'Tree');
% [CVTrainPred2, CVTrainScore2] = predict(Ensemble2{AgeGroup}, scoreCVTrain2);
% 
% disp('CV Training Complete....');
% %% CV Testing
% 
% [CVTestPred2, CVTestScore2] = predict(Ensemble2{AgeGroup},scoreCVTest2);
% disp('CV Testing Complete....');
% 
% FinalPredM2 = [FinalPredM2;CVTestPred2];
% %% Combine the 2 results
% CVTestScore1R = CVTestScore1(:,1)./max(abs(CVTestScore1(:,1)));
% CVTestScore2R = CVTestScore2(:,1)./max(abs(CVTestScore2(:,1)));
% CVTestScore12R = (CVTestScore1R + CVTestScore2R.*GISTFlag(AgeGroup)) < 0;
% % plot(CVTestScore1R,'b');
% % hold on;
% % plot(CVTestScore2R,'r');
% 
% FinalPredM12 = [FinalPredM12;CVTestScore12R];
% %% Display Accuracy
% 
% disp(['Train Accuracy Words ', num2str(sum(CVTrainPred1==CVTrainLabels)/sum(TrainIdxs))]);
% disp(['Test Accuracy Words ', num2str(sum(CVTestPred1==CVTestLabels)/sum(TestIdxs))]);
% 
% disp(['Train Accuracy GIST ', num2str(sum(CVTrainPred2==CVTrainLabels)/sum(TrainIdxs))]);
% disp(['Test Accuracy GIST ', num2str(sum(CVTestPred2==CVTestLabels)/sum(TestIdxs))]);
% 
% 
% disp(['Test Accuracy Combined ', num2str(sum(CVTestScore12R==CVTestLabels)/sum(TestIdxs))]);
% 
% end
% 
% 
% 
% disp(['Test Accuracy Model 1', num2str(sum(FinalPredM1(1:1000)==CVTestLabels)/NTest)]);
% disp(['Test Accuracy Model 2', num2str(sum(FinalPredM2(1:1000)==CVTestLabels)/NTest)]);
% disp(['Test Accuracy Combined', num2str(sum(FinalPredM12(1:1000)==CVTestLabels)/NTest)]);