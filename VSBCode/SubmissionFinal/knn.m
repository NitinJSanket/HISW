%% Instance-based Method {K-Nearest Neighbors}
function yHat = knn(trainX, trainY, testX)
    % Building the model
    model = fitcknn(trainX,trainY);

    % Predicting the test labels
    yHat = predict(model,testX);
end