%% Semi-supervised Dimensionality Reduction {PCA-ed data for Support Vector Machine}
function yHat = pcaSVM(trainX, trainY, testX)
    numTrainRows = size(trainX,1);
    
    % Semi-supervised PCA
    [~,score,~,~,expVar] = pca([trainX;testX]);
    cumsumExpVar = cumsum(expVar);
    minPC = min(find(cumsumExpVar>=99));
    trainXPCA = score(1:numTrainRows,1:minPC);
    testXPCA = score(numTrainRows+1:end,1:minPC);
    
    % Building the model
    model = fitcsvm(trainXPCA,trainY);

    % Predicting the test labels
    yHat = predict(model,testXPCA);
end