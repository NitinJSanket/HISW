%% Boiler Plate Code for Final Stretch
trainX = [mWordsTrain; mWordsTest];
imgFX = [mImageFeaturesTrain; mImageFeaturesTest];
trainY = [vGendersTrain; vGendersTest];
rankedX = mRankedFeatures;
actualWords = mWordsActual;
posTags = vPosTags;
wordLengths = vWordLengths;

%% Feature Selection/Reduction/Normalization/Standardization

numX = 1000

% allTrainX = cell(16,1);
% allTrainX{13} = [stemFeaturesWithRanking(trainX,actualWords,rankedX,numX),imgFX];
% allTrainX{14} = normFeatures([stemFeaturesWithRanking(trainX,actualWords,rankedX,numX),imgFX]);
% allTrainX{15} = standardizeFeatures([stemFeaturesWithRanking(trainX,actualWords,rankedX,numX),imgFX]);
% allTrainX{16} = standardizeFeatures(normFeatures([stemFeaturesWithRanking(trainX,actualWords,rankedX,numX),imgFX]));

featuresXStemRanked = stemFeaturesWithRanking(trainX,actualWords,rankedX,numX);
posX = createPOSFeatures(trainX, posTags);
wordFeatX = createWordFeatures(trainX, wordLengths);

allTrainX = cell(1,1);
allTrainX{1} = normFeatures(standardizeFeatures([featuresXStemRanked, imgFX, posX, wordFeatX]));

selTrainY = trainY;
trainAcc = zeros(size(allTrainX,1),1);
testAcc = zeros(size(allTrainX,1),1);
for iter = 1:size(allTrainX,1)
    selTrainX = allTrainX{iter};
    size(selTrainX,1)
    
    %Stratified CVPartition
    cvPartition = cvpartition(vGendersTrain,'Holdout',.2);
    heldInIndices = [training(cvPartition,1);true(size(vGendersTest,1),1)];
    heldOutIndices = test(cvPartition,1);

    heldInX = selTrainX(heldInIndices,:);
    heldInY = selTrainY(heldInIndices,:);

    heldOutX = selTrainX(heldOutIndices,:);
    heldOutY = selTrainY(heldOutIndices,:);

    bestModel = fitensemble(heldInX,heldInY,...
                 'RobustBoost',2000,'Tree',...
                 'nprint', 100);
    
%     errors = kfoldLoss(models,'mode','individual','lossfun','classiferror');
%     [~,modelIndex] = min(errors);
%     bestModel = models.Trained{modelIndex};
     
    trainAcc(iter) = mean(predict(bestModel,heldInX)==heldInY)
    testAcc(iter) = mean(predict(bestModel,heldOutX)==heldOutY)
end