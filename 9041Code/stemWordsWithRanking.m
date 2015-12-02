function [mFeaturesCollapsed] = stemWordsWithRanking(cWords, mFeatures, mRankedFeatures, iCutoffRank)

%% Select features till cutoffRank
[~, order] = sort(mRankedFeatures(:,2));
mRankedFeaturesSorted = mRankedFeatures(order,:);
vTopFeatures = mRankedFeaturesSorted(1:iCutoffRank,1);

%% Stemming each word in cWordsActual
cWordsStemmed = cell(size(cWords));
for iter = 1:length(cWords)
    cWordsStemmed{iter} = porterStemmer(cWords{iter});
end

%% Findind duplicate indices after stemming
[~,uniqueIndices] = unique(cWordsStemmed);
duplicateIndices = setdiff(1:length(cWordsStemmed), uniqueIndices)';

%% Collapsing Counts and Removing duplicates
mFeaturesCollapsed = mFeatures;
for iter= 1:length(duplicateIndices)
    % Find all common indices to this duplicate entry
    commonIndices = find(strcmp(cWordsStemmed, cWordsStemmed{duplicateIndices(iter)}));
    
    % Check if any of the commonIndices is in vTopFeatures
    if sum(ismember(commonIndices,vTopFeatures)) > 0
        % Sum all entries in the first common index and set all other common
        % indices to 0 so that they don't sum up again
        mFeaturesCollapsed(:,commonIndices(1)) = sum(mFeaturesCollapsed(:,commonIndices),2);
        mFeaturesCollapsed(:,commonIndices(2:end)) = 0;
        
        % Add the first common index to vTopFeatures, so as to prevent it
        % from getting deleted
        vTopFeatures(end+1) = commonIndices(1);
    end
end

% Set non-top feature indices to -1
vNotTopFeatures = setdiff(1:size(mRankedFeatures,1), vTopFeatures)';
mFeaturesCollapsed(:,vNotTopFeatures) = -1;

% Set duplicate indices to -1
mFeaturesCollapsed(:,duplicateIndices) = -1;

% Delete all -1 indices
mFeaturesCollapsed(:,find(mFeaturesCollapsed(1,:)==-1)) = [];

end
