%% Boiler Plate Code for Final Stretch
function [bestModel, trainAcc, testAcc] = boilerPlateCVModel(trainX, trainY, rankedX, actualWords)
    
    %% Feature Selection/Reduction/Normalization/Standardization
    
    % you can use following functions - 
    % standardizeFeatures,
    % freqRelFeatures, 
    % stemFeatures, 
    % stemFeaturesWithRanking
    % example: finalTrainX = stemWordsWithRanking(actualWords, trainX, rankedX, 3000);
    selTrainX = SOME_OF_THE_ABOVE_FUNCTIONS(trainX);

    %unless you need to change 0/1 labels to -1/1 or 1/2 or ...
    selTrainY = trainY; 

    %% Divide dataset into 2 parts - held-in for K-folds model, held-out for testing

    %Stratified CVPartition
    cvPartition = cvpartition(trainY,'Holdout',.20);
    heldInIndices = training(cvPartition,1);
    heldOutIndices = test(cvPartition,1);

    heldInX = selTrainX(heldInIndices,:);
    heldInY = selTrainY(heldInIndices,:);

    heldOutX = selTrainX(heldOutIndices,:);
    heldOutY = selTrainY(heldOutIndices,:);

    %% Build a 10-fold cross validated model on heldIn-indices and check error

    %Stratified CVPartition
    numFolds = 10;
    cvPartition = cvpartition(heldInY,'KFold',numFolds);
    allModels = cell(numFolds,1);
    cvTestErrors = zeros(numFolds,1);

    for iter = 1:numFolds
        kfTrainIndices = training(cvPartition,iter);
        kfTestIndices = test(cvPartition,iter);
        kfTrainX = heldInX(kfTrainIndices,:);
        kfTrainY = heldInY(kfTrainIndices);
        kfTestX = heldInX(kfTestIndices,:);
        kfTestY = heldInY(kfTestIndices);

        %CHANGE THIS PART ACCORDING TO YOUR MODEL
        allModels{iter} = mnrfit(kfTrainX,kfTrainY);
        allModels{iter}(isnan(allModels{iter})) = 0;

        % CHANGE THIS PART ACCORDING TO YOUR PREDICT FUNCTION
        [~,predicted] = max(mnrval(allModels{iter},kfTestX),[],2);
        cvTestErrors(iter) = mean(predicted~=kfTestY);
    end

    %% Select the Best Model
    [~,modelIndex] = min(cvTestErrors);
    bestModel = allModels{modelIndex};

    %% Calculate Held-in Train Accuracy and Held-out Test Accuracy

    % CHANGE THIS PART ACCORDING TO YOUR PREDICT FUNCTION
    [~,predicted] = max(mnrval(bestModel,heldInX),[],2);
    trainAcc = mean(predicted==heldInY);
    [~,predicted] = max(mnrval(bestModel,heldOutX),[],2);
    testAcc = mean(predicted==heldOutY);

end