clc
clear all
close all

addpath(genpath('./RF'));
addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');
addpath('./liblinear');

% addpath('./drtoolbox/', './drtoolbox/techniques/');
% rmpath('./drtoolbox/', './drtoolbox/techniques/');
%% Read Everything
TrainImgs = csvread([pwd, '/train/images_train.txt']);
TrainLabels = csvread([pwd, '/train/genders_train.txt']);
TrainWords = csvread([pwd, '/train/words_train.txt']);
TrainImgFeatures = csvread([pwd, '/train/image_features_train.txt']);

disp('Reading Data Complete....');
%% Split into CV Training and Testing Sets
NTest = 1000;
NTrain = length(TrainLabels)-NTest;
CVTrainImgs = TrainImgs(1:NTrain,:);
CVTestImgs = TrainImgs(NTrain+1:end, :);


CVTrainLabels = TrainLabels(1:NTrain,:);
CVTestLabels = TrainLabels(NTrain+1:end, :);

CVTrainWords = TrainWords(1:NTrain,:);
CVTestWords = TrainWords(NTrain+1:end, :);

% CVTrainWords = bsxfun(@rdivide, bsxfun(@minus, CVTrainWords, mean(CVTrainWords)), var(CVTrainWords) + 1e-10);
% CVTestWords = bsxfun(@rdivide, bsxfun(@minus, CVTestWords, mean(CVTestWords)), var(CVTestWords) + 1e-10);

CVTrainImgFeatures = TrainImgFeatures(1:NTrain,:);
CVTestImgFeatures = TrainImgFeatures(NTrain+1:end, :);

%% CV Training
% [dbn] = rbm(CVTrainWords);
% [new_feat, new_feat_test ] = newFeature_rbm(dbn,CVTrainWords,CVTestWords);
% [ precision_ae_log ] = logistic( new_feat, CVTrainLabels, new_feat_test, CVTestWords);

% Ensemble = fitensemble(X,Y,Method,NLearn,Learners)

% NB with PCA
% NCoeff = 3000;
% [coeffCVTrain,scoreCVTrain,latentCVTrain] = pca(CVTrainWords);
% NB = fitcnb(scoreCVTrain(:,1:NCoeff),CVTrainLabels);
% CVTrainPred = predict(NB,scoreCVTrain(:,1:NCoeff));
% disp(['Train Accuracy ', num2str(sum(CVTrainPred==CVTrainLabels)/NTrain)]);

% %%
% opts= struct;
% opts.depth= 18;
% opts.numTrees= 10;
% opts.numSplits= 100;
% opts.verbose= true;
% opts.classifierID = 1; % weak learners to use. Can be an array for mix of weak learners too
% 
% RFModel = forestTrain(CVTrainWords, CVTrainLabels, opts);
% CVTrainPred = forestTest(RFModel, CVTrainWords);
% 
%% CV Testing
% CVTestPred = forestTest(RFModel, CVTestWords);
% [coeffCVTest,scoreCVTest,latentCVTest] = pca(CVTestWords);
% % CVTestPred = predict(NB,scoreCVTest(:,1:NCoeff));
% [coeffTrain,scoreTrain,latentTrain] = pca(TrainWords);
% [coeffCVTest,scoreCVTest,latentCVTest] = pca(CVTestWords);
% NCoeff = 1000;

% NB = fitcnb(scoreTrain(1:NTrain,1:NCoeff),CVTrainLabels);

% CVTrainPred = predict(NB,scoreTrain(1:NTrain,1:NCoeff));
% CVTestPred = predict(NB,scoreTrain(NTrain+1:end,1:NCoeff));
% Ensemble = fitensemble(scoreTrain(1:NTrain,1:NCoeff),CVTrainLabels,'AdaBoostM1',100,'Trees');
% Ensemble = fitensemble(CVTrainWords,CVTrainLabels,'AdaBoostM1',800,'Tree');
% CVTrainPred = predict(Ensemble,CVTrainWords);
% CVTestPred = predict(Ensemble,CVTestWords);

% CVTrainPred = predict(Ensemble,scoreTrain(1:NTrain,1:NCoeff));

% CVTestPred = predict(Ensemble,scoreTrain(NTrain+1:end,1:NCoeff));

%% Display Accuracy
% SVMModel = fitcsvm(CVTrainImgFeatures,CVTrainLabels);
% CVTrainPred = predict(SVMModel,CVTrainImgFeatures);
% CVTestPred = predict(SVMModel,CVTestImgFeatures);

% Ensemble1 = fitensemble(CVTrainWords,CVTrainLabels,'AdaBoostM1',800,'Tree');
% [CVTrainPred1, CVTrainScore1] = predict(Ensemble1,CVTrainWords);
% [CVTestPred1, CVTestScore1] = predict(Ensemble1,CVTestWords);
% 
% Ensemble2 = fitensemble(CVTrainImgFeatures,CVTrainLabels,'AdaBoostM1',1000,'Tree');
% [CVTrainPred2, CVTrainScore2] = predict(Ensemble2,CVTrainImgFeatures);
% [CVTestPred2, CVTestScore2] = predict(Ensemble2,CVTestImgFeatures);
% 
% % disp(['Train Accuracy ', num2str(sum(CVTrainPred1==CVTrainLabels)/NTrain)]);
% % disp(['Test Accuracy ', num2str(sum(CVTestPred1(1:1000)==CVTestLabels)/NTest)]);
% % 
% % disp(['Train Accuracy ', num2str(sum(CVTrainPred2==CVTrainLabels)/NTrain)]);
% % disp(['Test Accuracy ', num2str(sum(CVTestPred2(1:1000)==CVTestLabels)/NTest)]);
% 
% 
% CVTrainPred1(CVTrainPred1==0) = -1;
% CVTrainPred2(CVTrainPred2==0) = -1;
% 
% CVTestPred1(CVTestPred1==0) = -1;
% CVTestPred2(CVTestPred2==0) = -1;
% 
% WM1 = 0.5;
% WM2 = 1-WM1;
% CVTrainPred12 = (CVTrainScore1(:,1)./std(CVTrainScore1(:,1)).*WM1 + CVTrainScore2(:,1)./std(CVTrainScore2(:,1)).*WM2)<0;
% CVTestPred12 = (CVTestScore1(:,1)./std(CVTestScore1(:,1)).*WM1 + CVTestScore2(:,1)./std(CVTestScore1(:,1)).*WM2)<0;
% 
% disp(['Train Accuracy ', num2str(sum(CVTrainPred12==CVTrainLabels)/NTrain)]);
% disp(['Test Accuracy ', num2str(sum(CVTestPred12(1:1000)==CVTestLabels)/NTest)]);


% Ensemble = fitensemble(TrainWords,TrainLabels,'AdaBoostM1',900,'Tree');
% TrainPred = predict(Ensemble,TrainWords);
% disp(['Train Accuracy ', num2str(sum(TrainPred==TrainLabels)/length(TrainLabels))]);

% TrainWords = csvread([pwd, '/test/words_test.txt']);


% TestImgs = csvread([pwd, '/test/images_test.txt']);
% TestWords = csvread([pwd, '/test/words_test.txt']);
% TestImgFeatures = csvread([pwd, '/test/image_features_test.txt']);
% 
% Ensemble1 = fitensemble(TrainWords,TrainLabels,'AdaBoostM1',1100,'Tree');
% [TrainPred1, TrainScore1] = predict(Ensemble1,TrainWords);
% 
% Ensemble2 = fitensemble(TrainImgFeatures,TrainLabels,'AdaBoostM1',1400,'Tree');
% [TrainPred2, TrainScore2] = predict(Ensemble2,TrainImgFeatures);
% 
% [TestPred1, TestScore1] = predict(Ensemble1,TestWords);
% [TestPred2, TestScore2] = predict(Ensemble2,TestImgFeatures);
% 
% 
% WM1 = 0.5;
% WM2 = 1-WM1;
% TrainPred12 = (TrainScore1(:,1)./std(TrainScore1(:,1)).*WM1 + TrainScore2(:,1)./std(TrainScore2(:,1)).*WM2)<0;
% 
% TestPred12 = (TestScore1(:,1)./std(TestScore1(:,1)).*WM1 + TestScore2(:,1)./std(TestScore2(:,1)).*WM2)<0;


% TestImgs = csvread([pwd, '/test/images_test.txt']);
% TestWords = csvread([pwd, '/test/words_test.txt']);
% TestImgFeatures = csvread([pwd, '/test/image_features_test.txt']);

[coeffCVTrain,scoreCVTrain,latentCVTrain] = pca(CVTrainWords);
NC = 100;
Ensemble1 = fitensemble(scoreCVTrain(1:NTrain,1:NC),CVTrainLabels,'AdaBoostM1',800,'Tree');
[CVTrainPred1, CVTrainScore1] = predict(Ensemble1,scoreCVTrain(1:NTrain,1:NC));

% Ensemble2 = fitctree(CVTrainImgFeatures,CVTrainLabels,'MaxNumSplits',2,'MaxNumCategories',20);
Ensemble2 = fitensemble(CVTrainImgFeatures,CVTrainLabels,'AdaBoostM1',100,'Tree');
% Ensemble2 = fitcecoc(CVTrainWords,CVTrainLabels);
% Ensemble2 = fitcsvm(CVTrainWords,CVTrainLabels);
[CVTrainPred2, CVTrainScore2] = predict(Ensemble2,CVTrainWords);

CVTrainPred2(CVTrainPred2==0) = -1;
figure,
plot(abs(CVTrainScore2(CVTestPred2==CVTestLabels,1)));

% [CVTestPred1, CVTestScore1] = predict(Ensemble1,CVTestWords);
[CVTestPred2, CVTestScore2] = predict(Ensemble2,scoreCVTrain(NTrain+1:end,1:NC));


disp(['Test Accuracy ', num2str(sum(CVTestPred2==CVTestLabels)/length(CVTestLabels))]);


WM1 = 0.5;
WM2 = 1-WM1;
CVTrainPred12 = (CVTrainScore1(:,1)./std(CVTrainScore1(:,1)).*WM1 + CVTrainScore2(:,1)./std(CVTrainScore2(:,1)).*WM2)<0;

CVTestPred12 = (CVTestScore1(:,1)./std(CVTestScore1(:,1)).*WM1 + CVTestScore2(:,1)./std(CVTestScore2(:,1)).*WM2)<0;

disp(['Train Accuracy ', num2str(sum(CVTrainPred12==CVTrainLabels)/length(CVTrainLabels))]);
disp(['Test Accuracy ', num2str(sum(CVTestPred2==CVTestLabels)/length(CVTestLabels))]);
% dlmwrite('submit.txt', TestPred12);