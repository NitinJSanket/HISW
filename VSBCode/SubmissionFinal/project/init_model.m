function models = init_model()

addpath('./liblinear');
addpath('./libsvm');
models = load('models.mat');

% Example:
% model.svmw = SVM.w;
% model.lrw = LR.w;
% model.classifier_weight = [c_SVM, c_LR];
