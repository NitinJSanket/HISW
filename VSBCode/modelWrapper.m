iCutoffRank = 2885;
[mFeaturesCollapsed] = stemWordsWithRanking(mWordsActual, mWordsTrain, ...
                            mRankedFeatures, iCutoffRank);

iKFolds = 2;
[vModels,vTrainAccuracy,vTestAccuracy] = createCVModels(mFeaturesCollapsed,...
                                            vGendersTrain, iKFolds);