function [ageModels,trainAccuracy] = createAgeModels(wordsX, wordsActual, rankedFeatures,...
                                    imageFeaturesX, labelsY,...
                                    wordsTest, imageFeaturesTest)
    ageX = imageFeaturesX(:,1);
    wordsStemX = stemWordsWithRanking(wordsActual, wordsX, rankedFeatures, 2885);
    
%     ageSplits = [0,10,13,17,20,23,27,30,33,37,40,43,47,50,53,57,60,63,67,70];
    ageSplits = [0,15,25,40,55,70];
    
    ageModels = cell(length(ageSplits)-1,1);
    correctPredictions = zeros(length(ageSplits)-1,1);
    
    for iter = 2:length(ageSplits)
        ageIndicesX = find(ageSplits(iter-1) <= ageX & ageX< ageSplits(iter));
        if length(ageIndicesX) > 0
            iter
            ageModels{iter-1} = fitcsvm(wordsStemX(ageIndicesX,:),...
                                            labelsY(ageIndicesX),...
                                            'KernelFunction','rbf'); 

            predictions = predict(ageModels{iter-1},wordsStemX(ageIndicesX,:));
            correctPredictions(iter-1) = sum(predictions == labelsY(ageIndicesX));
        end
    end
    
    trainAccuracy = sum(correctPredictions)/length(wordsX);
end