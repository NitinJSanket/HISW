vIdxMale = find(vGendersTrain==0);
vIdxFemale = find(vGendersTrain==1);

mWordsTrainMale = mWordsTrain(vIdxMale,:);
mWordsTrainFemale = mWordsTrain(vIdxFemale,:);

mWordsTrainMale = mean(mWordsTrainMale);
mWordsTrainFemale = mean(mWordsTrainFemale);

idxMale = find(mWordsTrainMale==0);
idxFemale = find(mWordsTrainFemale==0);

idxCommon = intersect(idxMale,idxFemale);

mWordsTrainMale(idxCommon)=[];
mWordsTrainFemale(idxCommon)=[];

mWordsTrainDiff = (abs(mWordsTrainMale-mWordsTrainFemale));

[mWordsTrainDiffSort, idx] = sort(mWordsTrainDiff,2,'descend');

vidxWordsFeatures = idx(1:3000);

figure;plot(1:size(mWordsTrainMale,2),mWordsTrainDiffSort)