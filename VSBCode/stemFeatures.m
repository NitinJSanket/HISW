function featuresStemmed = stemFeatures(wordsActual, featuresX)
    %% Stemming each word in wordsActual
    wordsStemmed = cell(size(wordsActual));
    for iter = 1:length(wordsActual)
        wordsStemmed{iter} = porterStemmer(wordsActual{iter});
    end

    %% Findind duplicate indices after stemming
    [~,uniqueIndices] = unique(wordsStemmed);
    duplicateIndices = setdiff(1:length(wordsStemmed), uniqueIndices)';

    %% Summing Counts and Removing Duplicates
    featuresStemmed = featuresX;
    for iter= 1:length(duplicateIndices)
        % Find all common indices to this duplicate entry
        commonIndices = find(strcmp(wordsStemmed, wordsStemmed{duplicateIndices(iter)}));

        % Sum all entries in the first common index and set all other common
        % indices to 0 so that they don't sum up again
        featuresStemmed(:,commonIndices(1)) = sum(featuresStemmed(:,commonIndices),2);
        featuresStemmed(:,commonIndices(2:end)) = 0;
    end

    % Delete duplicate indices
    featuresStemmed(:,duplicateIndices) = [];

end
