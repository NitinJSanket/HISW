ageTest = mImageFeaturesTest(:,1);
wordsStemTest = stemWordsWithRanking(mWordsActual, mWordsTest, mRankedFeatures, 2885);

% ageSplits = [0,10,15,20,25,30,35,40,45,50,55,60,65,70];
    ageSplits = [0,10,13,17,20,23,27,30,33,37,40,43,47,50,53,57,60,63,67,70];

predictions = zeros(length(wordsStemTest),1);

for iter = 2:length(ageSplits)
    ageIndicesTest = find(ageSplits(iter-1) <= ageTest & ageTest < ageSplits(iter));
    if length(ageIndicesTest) > 0
        predictions(ageIndicesTest) = predict(ageModels{iter-1},wordsStemTest(ageIndicesTest,:));
    end
end