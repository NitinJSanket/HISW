function [mWordsCollapsed] = stemWords(cWordsActual, mWords)
%% Stemming each word in cWordsActual
mWordsStemmed = cell(size(cWordsActual));
for iter = 1:length(cWordsActual)
    mWordsStemmed{iter} = porterStemmer(cWordsActual{iter});
end

%% Findind duplicate indices after stemming
[~,uniqueIndices] = unique(mWordsStemmed);
duplicateIndices = setdiff(1:length(mWordsStemmed), uniqueIndices)';

%% Collapsing Counts and Removing duplicates
mWordsCollapsed = mWords;
for iter= 1:length(duplicateIndices)
    % Find all common indices to this duplicate entry
    commonIndices = find(strcmp(mWordsStemmed, mWordsStemmed{duplicateIndices(iter)}));
    
    % Sum all entries in the first common index and set all other common
    % indices to 0 so that they don't sum up again
    mWordsCollapsed(:,commonIndices(1)) = sum(mWordsCollapsed(:,commonIndices),2);
    mWordsCollapsed(:,commonIndices(2:end)) = 0;
end

% Delete duplicate indices
mWordsCollapsed(:,duplicateIndices) = [];

