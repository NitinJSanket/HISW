clc
clear all
close all
warning off;

%% Add all necessary paths and packages
addpath(genpath('./RF'));
% addpath('./DL_toolbox/util','./DL_toolbox/NN','./DL_toolbox/DBN');
addpath('./liblinear');

addpath('../../GIST/');
addpath('../MIToolbox/');
addpath('../FEAST/');
addpath('./libsvm/');

%% Load Everything
load('GISTFeatures.mat');
load('GISTFeaturesT.mat');
load('Data.mat');
TrainWordsOrg = TrainWords;
load('WordsData.mat');
load('LIBGRSM.mat');
LIBGRSM = y';
load('LIBGRSMTest.mat');
LIBGRSMTest = y';
disp('Reading Data Complete....');

%% Initialize other stuff
TrainAcc = zeros(NIter,1);
TestAcc = zeros(NIter,1);