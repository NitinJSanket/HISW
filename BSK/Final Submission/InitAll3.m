clc
clear all
close all
warning off;

%% Add all necessary paths and packages
addpath(genpath('./RF'));
addpath('./liblinear');
addpath('./libsvm/');

%% Load Everything
load('GISTFeatures.mat');
load('GISTFeaturesT.mat');
load('Data.mat');
TrainWords = [TrainWords; TestWords];
load('WordsData.mat');
TestLabels = vGendersTest;
TrainLabels = [TrainLabels; TestLabels(1:2500)];
TrainImgFeatures = [TrainImgFeatures; TestImgFeatures(1:2500,:)];
TrainImgs = [TrainImgs; TestImgs(1:2500,:)];
TestLabels = TestLabels(2501:end);
TestImgFeatures = TestImgFeatures(2501:end,:);
TestImgs(2500:end,:) = TestImgs(2500:end,:);
CVTrainGISTFeatures = [GISTFeatures;GISTFeaturesT(1:2500,:)];
CVTestGISTFeatures = GISTFeaturesT(2501:end,:);
disp('Reading Data Complete....');

%% Initialize other stuff
TrainAcc = zeros(NIter,1);
TestAcc = zeros(NIter,1);
iter = 1;