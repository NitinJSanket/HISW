function predictions = make_final_prediction(models,XTest,XTrain)
    %% Load supporting data
    load('supportingData.mat')

    %% Train Features
    wordX_Tr = XTrain(:,1:5000);
    imgFX_Tr = XTrain(:,35001:35007);
    
    % Derived Train Features
    wordFX_Tr = createWordFeatures(wordX_Tr, vWordLengths);
    posX_Tr = createPOSFeatures(wordX_Tr, vPosTags);
    rank1KX_Tr = rankFeatures(wordX_Tr,mRankedFeatures,1000);
    hogX_Tr = createPixelFeatures(XTrain(:,5001:35000));
    
    %% Testing Features
    wordX_Ts = XTest(:,1:5000);
    imgFX_Ts = XTest(:,35001:35007);

    % Derived Test Features
    wordFX_Ts = createWordFeatures(wordX_Ts, vWordLengths);
    posX_Ts = createPOSFeatures(wordX_Ts, vPosTags);
    rank1KX_Ts = rankFeatures(wordX_Ts,mRankedFeatures,1000);
    stemRank1KX_Ts = stemFeaturesWithRanking(wordX_Ts,mWordsActual,mRankedFeatures,1000);
    stemRank3KX_Ts = stemFeaturesWithRanking(wordX_Ts,mWordsActual,mRankedFeatures,3000);
    hogX_Ts = createPixelFeatures(XTest(:,5001:35000));

    %% Logisitic Regression on 5k7 + HOG Norm
    lrX = normFeatures([wordX_Ts,imgFX_Ts,hogX_Ts]);
    lrY_Dummy = ones(size(XTest,1));
    [lrY, ~, ~] = predict(lrY_Dummy, sparse(lrX), models{1}, ['-q', 'col', '-b 1']);

    %% AdaBoost on 1k7+12+5+HOG Norm
    abX = normFeatures([rank1KX_Ts,imgFX_Ts,posX_Ts,wordFX_Ts,hogX_Ts]);
    [ab1Y, ~] = predict(models{2}, abX);

    %% AdaBoost on Stemmed 1k7+12+5+HOG Norm
    ab2X = normFeatures([stemRank1KX_Ts,imgFX_Ts,posX_Ts,wordFX_Ts,hogX_Ts]);
    [ab2Y, ~] = predict(models{3}, ab2X);
    
    %% AdaBoost on raw 5k7 Norm
    ab3X = normFeatures(wordX_Ts);
    [ab3Y, ~] = predict(models{4}, ab3X);
    
    %% RobustBoost on Stemmed 1k7+12+5+HOG Norm
    rb1X = normFeatures([stemRank1KX_Ts,imgFX_Ts,posX_Ts,wordFX_Ts,hogX_Ts]);
    [rb1Y, ~] = predict(models{5}, rb1X);

    %% RobustBoost on Stemmed 3k7+12+5+HOG Norm
    rb2X = normFeatures([stemRank3KX_Ts,imgFX_Ts,posX_Ts,wordFX_Ts,hogX_Ts]);
    [rb2Y, ~] = predict(models{6}, rb2X);
    
    %% LogitBoost Stemmed 1k7+12+5+HOG Norm
    lbX = normFeatures([stemRank1KX_Ts,imgFX_Ts,posX_Ts,wordFX_Ts,hogX_Ts]);
    [lbY, ~] = predict(models{7}, lbX);

    %% SVM-Intersection Kernel on 1k7+12+5+HOG Norm
    svm1X_Tr = normFeatures([rank1KX_Tr,imgFX_Tr,posX_Tr,wordFX_Tr,hogX_Tr]);
    svm1X_Ts = normFeatures([rank1KX_Ts,imgFX_Ts,posX_Ts,wordFX_Ts,hogX_Ts]);
    svm1K_Ts = kernel_intersection(svm1X_Tr, svm1X_Ts);

    svm1Y_Dummy = ones(size(svm1X_Ts,1));
    
    [svm1Y,~,~] = svmpredict(svm1Y_Dummy, [(1:size(svm1K_Ts,1))', svm1K_Ts], models{8});
    
    %% SVM-Linear Kernel on 1k7+HOG Norm
    svm2X_Tr = normFeatures([rank1KX_Tr,imgFX_Tr,hogX_Tr]);
    svm2X_Ts = normFeatures([rank1KX_Ts,imgFX_Ts,hogX_Ts]);
    svm2K_Ts =  kernel_poly(svm2X_Tr, svm2X_Ts, 1);
    
    svm2Y_Dummy = ones(size(svm2X_Ts,1));
    
    [svm2Y,~,~] = svmpredict(svm2Y_Dummy, [(1:size(svm2K_Ts,1))' svm2K_Ts], models.Model2{9});

    %% Predicting the final labels
    concatY = [lrY,ab1Y,rb1Y,lbY,svm1Y,rb2Y,svm2Y,ab2Y,ab3Y];
    Wts = [0.08, 0.15, 0.15, 0.08, 0.15, 0.18, 0.35, 0.15, 0.04];
    pred = sum(bsxfun(@times, Wts, concatY),2);

    predictions = pred > (0.45.*sum(Wts));  
end