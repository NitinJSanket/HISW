%% RobustBoost on Vidur's 1k Features
Model{1} = fitensemble(TrainWords,TrainLabels,'RobustBoost',2000,'Tree','RobustErrorGoal',0.1,'NPrint',10);
[PredTest1, PredTestScore1] = predict(Model{1}, TestWords);
[PredTrain1, PredTrainScore1] = predict(Model{1}, TrainWords);
disp('Model 1 Done....');

%% Adaboost M1 on Raw counts
Model{2} = fitensemble(TrainWordsOrg,TrainLabels,'AdaBoostM1',1000,'Tree','NPrint',10);
[PredTest2, PredTestScore2] = predict(Model{2}, TestWordsOrg);
[PredTrain2, PredTrainScore2] = predict(Model{2}, TrainWordsOrg);
disp('Model 2 Done....');

%% Adaboost M1 on Raw counts+Img Features (WTF!)
Model{3} = fitensemble([TrainWordsOrg, TrainImgFeatures],TrainLabels,'AdaBoostM1',1000,'Tree','NPrint',10);
[PredTest3, PredTestScore3] = predict(Model{3}, [TestWordsOrg, TestImgFeatures]);
[PredTrain3, PredTrainScore3] = predict(Model{3}, [TrainWordsOrg, TrainImgFeatures]);
disp('Model 3 Done....');

%% Liblinear's Logistic Regression L2 Regularized with auto C finder
LRArgs = 'col -b 1 -C -s 0';
Model{4} = train(TrainLabels, sparse(TrainWordsOrg), LRArgs);
[PredTest4, ~, PredTestScore4] = predict(ones(size(TestWords,1),1), sparse(TestWordsOrg), Model{4}, ['-q', 'col', '-b 1']);
[PredTrain4, ~, PredTrainScore4] = predict(TrainLabels, sparse(TrainWordsOrg), Model{4}, ['-q', 'col', '-b 1']);
disp('Model 4 Done....');

%% Matlab's Kernel Intersection SVM
Model{5} = fitcsvm(TrainWords,TrainLabels,'KernelFunction','kernel_intersection','OutlierFraction',0.1);
[PredTest5, PredTestScore5] = predict(Model{5}, TestWords);
[PredTrain5, PredTrainScore5] = predict(Model{5}, TrainWords);
disp('Model 5 Done....');

%% Image Features+1K
Model{6} = fitensemble([TrainWords,TrainImgFeatures],TrainLabels,'RobustBoost',2000,'Tree','RobustErrorGoal',0.1,'NPrint',10);
[PredTest6, PredTestScore6] = predict(Model{6}, [TestWords,TestImgFeatures]);
[PredTrain6, PredTrainScore6] = predict(Model{6}, [TrainWords,TrainImgFeatures]);
disp('Model 6 Done....');

%% LibSVM's Kernel Intersection SVM
kernel_intersection =  @(x,x2) kernel_intersection(x, x2);
[PredTrain7,PredTest7, Model{7}]= kernel_libsvm(TrainWords,TrainLabels,...
    TestWords,ones(size(TestWords,1),1),kernel_intersection);
disp('Model 7 Done....');

%% LibSVM's Kernel Intersection SVM on 5k+7Image Features
kernel_intersection =  @(x,x2) kernel_intersection(x, x2);
[PredTrain8,PredTest8, Model{8}]= kernel_libsvm([TrainWords,TrainImgFeatures],TrainLabels,...
    [TestWords,TestImgFeatures],ones(size(TestWords,1),1),kernel_intersection);
disp('Model 8 Done....');

%% LibSVM's Kernel Intersection SVM on 5k+7Image Features Normalized
kernel_intersection =  @(x,x2) kernel_intersection(x, x2);
FeatTrain = [TrainWords,TrainImgFeatures];
FeatTest = [TestWords,TestImgFeatures];
FeatTrainNormRows = sqrt(sum(abs(FeatTrain).^2,2));
FeatTrain = bsxfun(@times, FeatTrain, 1./FeatTrainNormRows);
FeatTestNormRows = sqrt(sum(abs(FeatTest).^2,2));
FeatTest = bsxfun(@times, FeatTest, 1./FeatTestNormRows);
[PredTrain9,PredTest9, Model{9}]= kernel_libsvm(FeatTrain,TrainLabels,...
    FeatTest,ones(size(TestWords,1),1),kernel_intersection);
disp('Model 9 Done....');