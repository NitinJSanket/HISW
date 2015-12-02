%% Initialize Everything
InitAll;

%% Vidur's Feature vector
TestWordsOrg = TestWords;
TestWords = stemWordsWithRanking(mWordsActual, TestWords, mRankedFeatures, 1000);

TrainWordsOrg = TrainWords;
TrainWords = stemWordsWithRanking(mWordsActual, TrainWords, mRankedFeatures, 1000);

%% Training
TrainForSubmission;

%% Combine
AllPredTrain = [PredTrain1,PredTrain2,PredTrain3,PredTrain4,PredTrain5,PredTrain6,PredTrain7,PredTrain8,PredTrain9];
AllPredTest = [PredTest1,PredTest2,PredTest3,PredTest4,PredTest5,PredTest6,PredCVTest7,PredCVTest8,PredCVTest9];

%% Lasso
Wts = [0.1422,0.0864,0.0906,0.1062,0.0351,0.1067,-0.0424,0.2003,0.2009];
PredTrainFinal = sum(bsxfun(@times, Wts, AllPredTrain),2);
PredTestFinal = sum(bsxfun(@times, Wts, AllPredTest),2);
PredTrainFinal = PredTrainFinal>0.463;
PredTestFinal = PredTestFinal>0.463;
dlmwrite('submit.txt', PredTestFinal);

    