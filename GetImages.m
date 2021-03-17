% make a folder structure to save cropped images
load AGC_Challenge3_Training.mat

pathImages = [pwd '/TRAINING/'];
pathTRAIN_DNN = [pwd '/TRAINING_DNN_GIVEN/'];

if ~exist(pathTRAIN_DNN, 'dir'), mkdir(pathTRAIN_DNN); end

for j = 1 : length( AGC_Challenge3_TRAINING )
    % exclude images with two face boxes
    if (AGC_Challenge3_TRAINING(j).id == -1), continue; end
    % exclude images with two face boxes
    if (size(AGC_Challenge3_TRAINING(j).faceBox, 1) >= 2)
        continue;
    elseif (size(AGC_Challenge3_TRAINING(j).faceBox, 1)==0)
        continue;
    end
    
    img_name = AGC_Challenge3_TRAINING(j).imageName;
    
    img = imread( sprintf('%s%s', pathImages, img_name) );
    
    % get bounding box and convert it
    img_bbox = AGC_Challenge3_TRAINING(j).faceBox;
    bbox_conv = [img_bbox(:,1:2) (img_bbox(:,3) - img_bbox(:,1)) (img_bbox(:,4) - img_bbox(:,2))];
    
    % cut picture to face
    img_crop = imcrop(img, bbox_conv);
    
    folder = [pathTRAIN_DNN num2str(AGC_Challenge3_TRAINING(j).id) '/'];
    if ~exist(folder, 'dir'), mkdir(folder); end
    
    % save image to new folder
    imwrite(img_crop, [folder img_name]);
end