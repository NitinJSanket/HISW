numImages = size(images_train,1);

for iter=1:numImages
    if vGendersTrain(iter)==0
        imwrite(uint8(reshape(images_train(iter,:),[100 100 3])),sprintf('%d.jpg',iter));
    end
    
end
