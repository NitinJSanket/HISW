clear all;
close all;

load XTest.mat
load XTrain.mat

tm = tic;
models = init_model();
toc(tm)

tp = tic;
predictions = make_final_prediction(models, XTest, XTrain);
toc(tp) 