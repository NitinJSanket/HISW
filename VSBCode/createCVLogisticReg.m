wordsX = mWordsTrain;
wordsXCollapsed = stemWordsWithRanking(mWordsActual, wordsX, mRankedFeatures, 3000);
derivedX = mImageFeaturesTrain;
gendersY = vGendersTrain;

% mnrfit needs labels to be positive integers
gendersY(gendersY==1) = 2;
gendersY(gendersY==0) = 1;


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
numFolds = 10;
cvPartition = cvpartition(heldInY,'KFold',numFolds);
models = cell(numFolds,1);
errors = zeros(numFolds,1);

for iter = 1:numFolds
    kfTrainIndices = training(cvPartition,iter);
    kfTestIndices = test(cvPartition,iter);
    kfTrainX = heldInXCollapsed(kfTrainIndices,:);
    kfTrainY = heldInY(kfTrainIndices);
    kfTestX = heldInXCollapsed(kfTestIndices,:);
    kfTestY = heldInY(kfTestIndices);
    
    models{iter} = mnrfit(kfTrainX,kfTrainY);
    models{iter}(isnan(models{iter})) = 0;
    
    [~,predicted] = max(mnrval(models{iter},kfTestX),[],2);
    errors(iter) = mean(predicted~=kfTestY);
end

%% Find the best model and give held-in train accuracy and held-out test accuracy
[~,modelIndex] = min(errors);
model = models{modelIndex};

[~,predicted] = max(mnrval(model,heldInXCollapsed),[],2);
heldInTrainAcc = mean(predicted==heldInY);
[~,predicted] = max(mnrval(model,heldOutXCollapsed),[],2);
heldOutTestAcc = mean(predicted==heldOutY);