%% Boiler Plate Code for Final Stretch
trainX = mWordsTrain;
imgFX = mImageFeaturesTrain;
trainY = vGendersTrain;
rankedX = mRankedFeatures;
actualWords = mWordsActual;
    
%% Feature Selection/Reduction/Normalization/Standardization

% you can use following functions - 
% standardizeFeatures,
% freqRelFeatures, 
% stemFeatures, 
% stemFeaturesWithRanking
% example: allTrainX{1} = stemWordsWithRanking(actualWords, trainX, rankedX, 3000);

% Just change this numX from 5000 to 3000 to 1000
% Life is Golden !!!
numX = 1000;

allTrainX = cell(16,1);
allTrainX{1} = rankFeatures(trainX,rankedX,numX);
allTrainX{2} = normFeatures(rankFeatures(trainX,rankedX,numX));
allTrainX{3} = standardizeFeatures(rankFeatures(trainX,rankedX,numX));
allTrainX{4} = standardizeFeatures(normFeatures(rankFeatures(trainX,rankedX,numX)));
allTrainX{5} = stemFeaturesWithRanking(trainX,actualWords,rankedX,numX);
allTrainX{6} = normFeatures(stemFeaturesWithRanking(trainX,actualWords,rankedX,numX));
allTrainX{7} = standardizeFeatures(stemFeaturesWithRanking(trainX,actualWords,rankedX,numX));
allTrainX{8} = standardizeFeatures(normFeatures(stemFeaturesWithRanking(trainX,actualWords,rankedX,numX)));
allTrainX{9} = [rankFeatures(trainX,rankedX,numX),imgFX];
allTrainX{10} = normFeatures([rankFeatures(trainX,rankedX,numX),imgFX]);
allTrainX{11} = standardizeFeatures([rankFeatures(trainX,rankedX,numX),imgFX]);
allTrainX{12} = standardizeFeatures(normFeatures([rankFeatures(trainX,rankedX,numX),imgFX]));
allTrainX{13} = [stemFeaturesWithRanking(trainX,actualWords,rankedX,numX),imgFX];
allTrainX{14} = normFeatures([stemFeaturesWithRanking(trainX,actualWords,rankedX,numX),imgFX]);
allTrainX{15} = standardizeFeatures([stemFeaturesWithRanking(trainX,actualWords,rankedX,numX),imgFX]);
allTrainX{16} = standardizeFeatures(normFeatures([stemFeaturesWithRanking(trainX,actualWords,rankedX,numX),imgFX]));

selTrainY = trainY;
trainAcc = zeros(size(allTrainX,1),1);
testAcc = zeros(size(allTrainX,1),1);
for iter = 1:size(allTrainX,1)
    selTrainX = allTrainX{iter};
    
    %Stratified CVPartition
    cvPartition = cvpartition(trainY,'Holdout',.20);
    heldInIndices = training(cvPartition,1);
    heldOutIndices = test(cvPartition,1);

    heldInX = selTrainX(heldInIndices,:);
    heldInY = selTrainY(heldInIndices,:);

    heldOutX = selTrainX(heldOutIndices,:);
    heldOutY = selTrainY(heldOutIndices,:);

    model = fitensemble(heldInX,heldInY,...
                 'RobustBoost',100,'Tree', 'nprint', 100);
    
    trainAcc(iter)  = mean(predict(model,heldInX)==heldInY);
    testAcc(iter)  = mean(predict(model,heldOutX)==heldOutY);
end