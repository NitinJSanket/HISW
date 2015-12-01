wordsX = mWordsTrain;
wordsXCollapsed = normalizeFeatures(stemWordsWithRanking(mWordsActual, wordsX, mRankedFeatures, 1000));
derivedX = mImageFeaturesTrain;
gendersY = vGendersTrain;

%% Divide dataset into 2 parts - held-in for K-folds model, held-out for testing
cvPartition = cvpartition(gendersY,'Holdout',.20);
heldInIndex = training(cvPartition,1);
heldOutIndex = test(cvPartition,1);

heldInX = wordsX(heldInIndex,:);
heldInXCollapsed = wordsXCollapsed(heldInIndex,:);
heldInY = gendersY(heldInIndex,:);

heldOutX = wordsX(heldOutIndex,:);
heldOutXCollapsed = wordsXCollapsed(heldOutIndex,:);
heldOutY = gendersY(heldOutIndex,:);

%% Build a 10-fold cross validated model on heldIn-indices and check error
models = fitensemble(heldInXCollapsed,heldInY,...
                 'RobustBoost',2000,'Tree',...
                 'kfold', 10,'NPrint',10);

errors = kfoldLoss(models,'mode','individual','lossfun','classiferror');

%% Find the best model and give held-in train accuracy and held-out test accuracy
[~,modelIndex] = min(errors);
model = models.Trained{modelIndex};

heldInTrainAcc = mean(predict(model,heldInXCollapsed)==heldInY);
heldOutTestAcc = mean(predict(model,heldOutXCollapsed)==heldOutY);


