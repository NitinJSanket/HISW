function predictions = make_final_prediction(model,XTest,XTrain)

load('supportingData.mat')

XTestWords = XTest(:,1:5000);
XTestImages = XTest(:,5001:35000);
XTestImageFeatures = XTest(:,35001:35007);

XTrainWords = XTrain(:,1:5000);
XTrainImages = XTrain(:,5001:35000);
XTrainImageFeatures = XTrain(:,35001:35007);

%% Logisitic Regression on 5k7 + GIST
XTestLR = double([XTestWords,XTestImageFeatures,XTestGISTFeatures]);
YTestLR = ones(size(XTest,1));
LRArgs = 'col -b 1 -C -s 0';
[testLabelsLR, ~, ~] = predict(YTestLR, sparse(XTestLR), model.modelLR, ['-q', 'col', '-b 1']);
%% AdaBoost on 1k7+12+5 Norm + GIST
XTestAB = normFeatures([...
    rankFeatures(XTestWords,mRankedFeatures,1000),...
    XTest(:,XTestImageFeatures),...
    createPOSFeatures(XTestWords, vPosTags),...
    createWordFeatures(XTestWords, vWordLengths), XTestGISTFeatures]);
[testLabelsAB, ~] = predict(model.model2, XTestAB);

%% RobustBoost on 1k7+12+5 Norm + GIST
XTestRB = normFeatures([...
    stemFeaturesWithRanking(XTestWords,mWordsActual,mRankedFeatures,1000),...
    XTestImageFeatures;...
    createPOSFeatures(XTestWords, vPosTags),...
    createWordFeatures(XTestWords, vWordLengths),XTestGISTFeatures]);

[testLabelsRB, ~] = predict(model.model3, XTestRB);
%% LogitBoost
XTestLB = normFeatures([...
    stemFeaturesWithRanking(XTestWords,mWordsActual,mRankedFeatures,1000),...
    XTestImageFeatures,...
    createPOSFeatures(XTestWords, vPosTags),...
    createWordFeatures(XTestWords, vWordLengths), XTestGISTFeatures]);

[testLabelsLB, ~] = predict(model.model4, XTestLB);
%% SVM-Intersection Kernel on Norm on 1k7+12+5
XTrainSVM = normFeatures([rankFeatures(XTrainWords,mRankedFeatures,1000), XTrainImageFeatures,...
    createPOSFeatures(XTrainWords, vPosTags),...
    createWordFeatures(XTrainWords, vWordLengths), XTrainGISTFeatures]);
YTestSVM = ones(size(XTestSVM,1));
XTestSVM = normFeatures([rankFeatures(XTestWords,mRankedFeatures,1000), XTestImageFeatures,...
    createPOSFeatures(XTestWords, vPosTags),...
    createWordFeatures(XTestWords, vWordLengths), XTestGISTFeatures]);
YTestSVM = ones(size(XTestSVM,1));
KTest = kernel_intersection(XTrainSVM, XTestSVM);
[testLabelsSVM,accTest,valsTest] = svmpredict(YTestSVM, [(1:size(KTest,1))' KTest], model.model5);

%%Predicting the final labels
testLabels = mode(testLabelsLR,testLabelsAB,testLabelsRB,testLabelsLB,testLabelsSVM,2);

predictions = testLabels;