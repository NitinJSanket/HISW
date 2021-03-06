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
% TrainImgs = csvread([pwd, '/train/images_train.txt']);
% TrainLabels = csvread([pwd, '/train/genders_train.txt']);
% TrainWords = csvread([pwd, '/train/words_train.txt']);
% TrainImgFeatures = csvread([pwd, '/train/image_features_train.txt']);
% 
% TestImgs = csvread([pwd, '/test/images_test.txt']);
% TestWords = csvread([pwd, '/test/words_test.txt']);
% TestImgFeatures = csvread([pwd, '/test/image_features_test.txt']);
load('Data.mat');

disp('Reading Data Complete....');

NIter = 5;

TrainAcc = zeros(NIter,1);
TestAcc = zeros(NIter,1);

%% Submission Code
% Model1 = fitcsvm(TrainWords,TrainLabels);
% Model1 = fitensemble(TrainWords,TrainLabels,'AdaBoostM1',800,'Tree');
% Model2 = fitensemble(GISTFeatures,TrainLabels,'AdaBoostM1',300,'Tree');
%
% [PredTest1, PredTestScore1] = predict(Model1, TestWords);
% [PredTest2, PredTestScore2] = predict(Model2, GISTFeaturesT);

% SSIdxs = load('TrainWords.mat');
% SSIdxs = SSIdxs.vidxWordsFeatures;
% NF = 3000;
%
% Model1 = fitensemble(TrainWords(:,SSIdxs(1:NF)),TrainLabels,'LogitBoost',800,'Tree');
% Model2 = train(TrainLabels, sparse(TrainWords(:,SSIdxs(1:NF))), ['-s 0', 'col', '-b 1']);
%
% [PredTest1, PredTestScore1] = predict(Model1, TestWords(:,SSIdxs(1:NF),1));
% [PredTest2, ~, PredTestScore2] = predict(ones(size(TestWords,1),1), sparse(TestWords(:,SSIdxs(1:NF))), Model2, ['-q', 'col', '-b 1']);
%
%
% PredTestScore1 = PredTestScore1(:,1)./max(abs(PredTestScore1(:,1)));
% PredTestScore2 = PredTestScore2(:,1)./max(abs(PredTestScore2(:,1)));
% PredTest = (PredTestScore1 + PredTestScore2) < 0;
% dlmwrite('submit.txt', PredTest);
%
% [PredTrain1, PredTrainScore1] = predict(Model1, TrainWords(:,SSIdxs(1:NF)));
% [PredTrain2, ~, PredTrainScore2] = predict(TrainLabels, sparse(TrainWords(:,SSIdxs(1:NF))), Model2, ['-q', 'col', '-b 1']);
% PredTrainScore1 = PredTrainScore1(:,1)./max(abs(PredTrainScore1(:,1)));
% PredTrainScore2 = PredTrainScore2(:,1)./max(abs(PredTrainScore2(:,1)));
% PredTrain = (PredTrainScore1 + PredTrainScore2) < 0;
%
% AccuracyTrain = sum(PredTrain==TrainLabels)/length(TrainLabels).*100;
%
% disp(['Train Accuracy ', num2str(AccuracyTrain)]);
%%
%% Feature Selection based on average word usage difference
SSIdxs = load('TrainWords.mat');
SSIdxs = SSIdxs.vidxWordsFeatures;
% SSIdxs = 1:5000;
NF = 3000;
M2Flag = 1;
LRArgs = 'col -b 1 -C -s 0'; %-s 0

clear PredScoreSaved1 PredScoreSaved2 TestSaved

tic
for iter = 2
    disp(['Executing ', num2str(iter), ' iteration out of ', num2str(NIter), ' iterations']);
    %% Split into CV Training and Testing Sets
    NTest = 1000;
    NTrain = length(TrainLabels)-NTest;
    
    AllIdxs = 1:length(TrainLabels);
    %     CVTestIdxs = randperm(length(TrainLabels), NTest);
    %     CVTrainIdxs = setdiff(AllIdxs, CVTestIdxs);
    if(iter==NIter)
        CVTestIdxs = 4001:size(TrainLabels);
    else
        CVTestIdxs = 1000*(iter-1)+1:1000*iter;
    end
    CVTrainIdxs = setdiff(AllIdxs, CVTestIdxs);
    
    
    CVTrainImgs = TrainImgs(CVTrainIdxs, SSIdxs(1:NF));
    CVTestImgs = TrainImgs(CVTestIdxs, SSIdxs(1:NF));
    
    CVTrainLabels = TrainLabels(CVTrainIdxs, :);
    CVTestLabels = TrainLabels(CVTestIdxs, :);
    
    CVTrainWords = TrainWords(CVTrainIdxs, SSIdxs(1:NF));
    CVTestWords = TrainWords(CVTestIdxs, SSIdxs(1:NF));
    
    CVTrainImgFeatures = TrainImgFeatures(CVTrainIdxs, :);
    CVTestImgFeatures = TrainImgFeatures(CVTestIdxs, :);
    
    %% Feature Selection
    CVTrainWordsSel = CVTrainWords;
    CVTestWordsSel = CVTestWords;
    
    CVTrainGISTSel = GISTFeatures(CVTrainIdxs, :);
    CVTestGISTSel = GISTFeatures(CVTestIdxs, :);
    
    %% Adaboost
    Model1 = fitensemble(CVTrainWordsSel,CVTrainLabels,'LogitBoost',800,'Tree');
    %     Model2 = fitensemble(CVTrainGISTSel,CVTrainLabels,'AdaBoostM1',300,'Tree');
    
    %% Linear SVM
    if(M2Flag)
        % Model1 = fitcsvm(CVTrainWordsSel,CVTrainLabels,'KernelFunction','histogramIntersection','Standardize',true);%,'Alpha',0.01*ones(size(CVTrainWords,1),1));
        disp('LR Active');
%          Model2 = train(CVTrainLabels, sparse(CVTrainWordsSel), 'col -b 1');
        Model2 = train(CVTrainLabels, sparse(CVTrainWordsSel), LRArgs);
%         Model2 = train(CVTrainLabels, sparse(CVTrainWordsSel), sprintf('-c %f -s 0', Model2(1))); % use the same solver: -s 0
    end
    %% Get Train and Test Error
    % If using Matlab's inbuilt models
    [PredCVTrain1, PredCVTrainScore1] = predict(Model1, CVTrainWordsSel);
    [PredCVTest1, PredCVTestScore1] = predict(Model1, CVTestWordsSel);
    
    if(M2Flag)
        % If using LibSVM
        [PredCVTrain2, ~, PredCVTrainScore2] = predict(CVTrainLabels, sparse(CVTrainWordsSel), Model2, ['-q', 'col', '-b 1']);
        [PredCVTest2, ~, PredCVTestScore2] = predict(CVTestLabels, sparse(CVTestWordsSel), Model2, ['-q', 'col', '-b 1']);
    end
    
    % [PredCVTrain2, PredCVTrainScore2] = predict(Model2, CVTrainGISTSel);
    % [PredCVTest2, PredCVTestScore2] = predict(Model2, CVTestGISTSel);
    
    if(M2Flag)
        PredCVTrainScore1 = PredCVTrainScore1(:,1)./max(abs(PredCVTrainScore1(:,1)));
        PredCVTrainScore2 = PredCVTrainScore2(:,1)./max(abs(PredCVTrainScore2(:,1)));
        PredCVTrain = (PredCVTrainScore1(:,1) + PredCVTrainScore2(:,1)) < 0;
    else
        PredCVTrain = PredCVTrain1;
    end
    
    if(M2Flag)
        PredCVTestScore1 = PredCVTestScore1(:,1)./max(abs(PredCVTestScore1(:,1)));
        PredCVTestScore2 = PredCVTestScore2(:,1)./max(abs(PredCVTestScore2(:,1)));
        PredCVTest = (PredCVTestScore1(:,1) + PredCVTestScore2(:,1)) < 0;
    else
        PredCVTest = PredCVTest1;
    end
    
    
    AccuracyCVTrain = sum(PredCVTrain==CVTrainLabels)/length(CVTrainLabels).*100;
    AccuracyCVTest = sum(PredCVTest==CVTestLabels)/length(CVTestLabels).*100;
    
    %% Display Results
    disp(['Train Accuracy Boost ', num2str(sum(PredCVTrain1==CVTrainLabels)/length(CVTrainLabels).*100)]);
    disp(['Test Accuracy Boost ', num2str(sum(PredCVTest1==CVTestLabels)/length(CVTestLabels).*100)]);
    
    disp(['Train Accuracy LR ', num2str(sum(PredCVTrain2==CVTrainLabels)/length(CVTrainLabels).*100)]);
    disp(['Test Accuracy LR ', num2str(sum(PredCVTest2==CVTestLabels)/length(CVTestLabels).*100)]);
    
    disp(['Train Accuracy Both ', num2str(AccuracyCVTrain)]);
    disp(['Test Accuracy Both ', num2str(AccuracyCVTest)]);
    TrainAcc(iter) = AccuracyCVTrain;
    TestAcc(iter) = AccuracyCVTest;
    PredScoreSaved1{iter} = PredCVTestScore1(:,1);
    PredScoreSaved2{iter} = PredCVTestScore2(:,1);
    TestSaved{iter} = CVTestLabels;
end
toc

%% Avg Results
AvgTrainAcc = mean(TrainAcc);
AvgTestAcc = mean(TestAcc);

disp(['Mean Train Accuracy ', num2str(AvgTrainAcc)]);
disp(['Mean Test Accuracy ', num2str(AvgTestAcc)]);

% %% Sarathba's test
% load('idxTest.mat');
% load('idxTrain.mat');
% CVTrainLabels = TrainLabels(idxTrain, :);
% CVTrainWords = TrainWords(idxTrain, :);
% CVTrainWordsSel = CVTrainWords;
%
% Model1 = fitensemble(CVTrainWordsSel,CVTrainLabels,'LogitBoost',800,'Tree');
% Model2 = train(CVTrainLabels, sparse(CVTrainWordsSel), ['-s 0', 'col', '-b 1']);
%
% [PredCVTrain1, PredCVTrainScore1] = predict(Model1, CVTrainWordsSel);
% [PredCVTrain2, ~, PredCVTrainScore2] = predict(CVTrainLabels, sparse(CVTrainWordsSel), Model2, ['-q', 'col', '-b 1']);
% PredCVTrainScore1 = PredCVTrainScore1(:,1)./max(abs(PredCVTrainScore1(:,1)));
% PredCVTrainScore2 = PredCVTrainScore2(:,1)./max(abs(PredCVTrainScore2(:,1)));
% PredCVTrain = (PredCVTrainScore1 - PredCVTrainScore2) < 0;
% AccuracyCVTrain = sum(PredCVTrain==CVTrainLabels)/length(CVTrainLabels).*100;
%
%
% CVTestLabels = TrainLabels(idxTest, :);
% CVTestWords = TrainWords(idxTest, :);
% CVTestWordsSel = CVTestWords;
% [PredCVTest1, PredCVTestScore1] = predict(Model1, CVTestWordsSel);
% [PredCVTest2, ~, PredCVTestScore2] = predict(CVTestLabels, sparse(CVTestWordsSel), Model2, ['-q', 'col', '-b 1']);
% PredCVTestScore1 = PredCVTestScore1(:,1)./max(abs(PredCVTestScore1(:,1)));
% PredCVTestScore2 = PredCVTestScore2(:,1)./max(abs(PredCVTestScore2(:,1)));
% PredCVTest = (PredCVTestScore1 - PredCVTestScore2) < 0;
% AccuracyCVTest = sum(PredCVTest==CVTestLabels)/length(CVTestLabels).*100;
%
% disp(['Mean Train Accuracy ', num2str(AccuracyCVTrain)]);
% disp(['Mean Test Accuracy ', num2str(AccuracyCVTest)]);
% % %%