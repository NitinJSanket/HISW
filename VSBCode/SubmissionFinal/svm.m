%% Discriminative Method {Support Vector Machine}
function yHat = svm(trainX, trainY, testX)
    % Building the model
    model = fitcsvm(trainX,trainY);

    % Predicting the test labels
    yHat = predict(model,testX);
end