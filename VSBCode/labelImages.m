close all;

images = images_test;
numImages = size(images,1);

f = figure; 
set(f,'KeyPressFcn',@KeyPressFcn) ;

% labels = zeros(numImages,1);
errorIndex = [3037,3040,3059,3159,3174,3204,3297];
for iter = errorIndex
    iter-2
    labels(iter-2)
    imshow(uint8(reshape(images(iter-2,:),[100 100 3])));
    [~,~,button] = ginput(1);
    labels(iter-2) = button; 
end
