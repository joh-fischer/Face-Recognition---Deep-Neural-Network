pathCelebs = [pwd '/SCRAPED/'];
pathTraining = [pwd '/TRAINING_SCRAPED/'];

if ~exist(pathTraining, 'dir'), mkdir(pathTraining); end

% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();
faceDetector.MergeThreshold = 12;

% add images from celebs folder
img_number = 1;
for j = 79 : 79
    
    targetFolder = [pathTraining num2str(j) '/'];
    if ~exist(targetFolder, 'dir'), mkdir(targetFolder); end
    
    folder = [pathCelebs num2str(j) '/'];    
    files = dir([folder '*.jpg']);
    
    disp(['Identity ' num2str(j) ' of ' num2str(80)]);
    
    for file = files'
        % check if is old image
        if regexp(file.name, regexptranslate('wildcard','image_A*.jpg')), continue; end
        
        % load image
        try
            img = imread( [folder file.name] );
        catch
           continue;
        end
        % Run the face detector.
        detected_bbox = faceDetector(img);
        
        % if no face was found, continue
        if isempty(detected_bbox), continue; end
        
        temp_bbox = detected_bbox;
        % cut if more than 2 faces were found
        while (size(temp_bbox, 1) > 2)
            [~, idx] = max(temp_bbox(:,1));
            temp_bbox(idx,:) = [];
        end
        
        for iBbox = 1:size(temp_bbox, 1)
            bbox_conv = temp_bbox(iBbox,:);
            % cut picture to face
            img_crop = imcrop(img, bbox_conv);
            
            img_name = ['image_C' num2str(img_number, '%05.f') '.jpg'];
            % save image to new folder
            imwrite(img_crop, [targetFolder img_name]);
            
            img_number = img_number + 1;
        end
    end
end

%save('AGC_TRAINING.mat', 'AGC_TRAINING');