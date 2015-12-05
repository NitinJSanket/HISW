function duplicateIndices = findStemWords(wordsActual)
    %% Stemming each word in wordsActual
    wordsStemmed = cell(size(wordsActual));
    for iter = 1:length(wordsActual)
        wordsStemmed{iter} = porterStemmer(wordsActual{iter});
    end

    %% Findind duplicate indices after stemming
    [~,uniqueIndices] = unique(wordsStemmed);
    duplicateIndices = setdiff(1:length(wordsStemmed), uniqueIndices)';

end
