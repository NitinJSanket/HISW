%% Train Classifier
negativeImgDir = fullfile('G:\Academics\Fall2015\ML520\Project\kit\kit\HISW\VSBCode\FemaleImages');
positiveImgDir = fullfile('G:\Academics\Fall2015\ML520\Project\kit\kit\HISW\VSBCode\MaleImages');
addpath(positiveImgDir);

% 0 is males, 1 is females
positiveClass = 0;
negativeClass = 1;

numImages = size(images_train,1);
positiveCounter = 1;
for iter=1:numImages
    if vGendersTrain(iter) == positiveClass
        s = struct('imageFilename',sprintf('%d.jpg',iter),'objectBoundingBoxes',[1,1,100,100]);
        positiveImgStruct(positiveCounter) = s; 
        positiveCounter = positiveCounter+1;
    end
end

trainCascadeObjectDetector('FemaleClassifier.xml',...
                            positiveImgStruct,negativeImgDir,...
                            'FalseAlarmRate',0.1,'NumCascadeStages',20);
                        
%% Use Classifier for Detection
detector = vision.CascadeObjectDetector('MaleClassifier.xml');
predictedMales = zeros(numImages,1);

for iter=1:numImages
        im = uint8(reshape(images_train(iter,:),[100 100 3]));
        bbox = step(detector,im);
        if size(bbox,1) > 0
            predictedMales(iter) = positiveClass;
%             detectedImg = insertObjectAnnotation(im,'rectangle',bbox,'male');
%             imshow(detectedImg);
        else
            predictedMales(iter) = negativeClass;
        end
        
            
end

accuracy = mean(predictedMales==vGendersTrain)
positivePred = length(predictedMales(predictedMales == positiveClass))
totalPositive = length(vGendersTrain(vGendersTrain == positiveClass))
negativePred = length(predictedMales(predictedMales == negativeClass))
negativePredCorrect = sum(predictedMales == negativeClass & vGendersTrain == negativeClass)
totalNegative = length(vGendersTrain(vGendersTrain == negativeClass))
