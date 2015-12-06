%% Generative Method {Naive Bayes}
function yHat = nb(trainX, trainY, testX)
    % Building the model
    model = fitcnb(trainX,trainY,'distribution','mvmn');
    
    % Predicting the test labels
    yHat = predict(model,testX);
end