function [mFeaturesNormalized] = normalizeFeatures(mFeatures)
    iNumSamples = size(mFeatures,1);
    mMeanFeatures = repmat(mean(mFeatures),[iNumSamples,1]);
    mStdDevFeatures = repmat(std(mFeatures),[iNumSamples,1]);
    
    mFeaturesNormalized = (mFeatures-mMeanFeatures)./mStdDevFeatures;
    mFeaturesNormalized(isnan(mFeaturesNormalized)) = 0;
end
