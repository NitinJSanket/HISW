function pixelX = createPixelFeatures(imageX)
    numImages = size(imageX,1);

    pixelX = zeros(numImages,576);
    for iter=1:numImages
        img = uint8(reshape(imageX(iter,:),[100 100 3]));
        pixelX(iter,:) = extractHOGFeatures(img,'cellsize',[16 16],'blocksize',[4 4]);
    end
end