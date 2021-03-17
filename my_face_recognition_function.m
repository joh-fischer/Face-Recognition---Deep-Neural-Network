function resultID = my_face_recognition_function(img, model)

threshold = 0.6;
resultID = -1;

% Create a cascade detector object.
faceDetector = vision.CascadeObjectDetector();
faceDetector.MergeThreshold = 12;

% Run the face detector.
detected_bbox = faceDetector(img);

% if no face was found, return empty array
if isempty(detected_bbox), return; end
 
% cut if more than 2 faces were found
temp_bbox = detected_bbox;
while (size(temp_bbox, 1) > 2)
    [~, idx] = max(temp_bbox(:,1));
    temp_bbox(idx,:) = [];
end

for iBox = 1:size(temp_bbox, 1)
    bbox = temp_bbox(iBox, :);
    
    % cut picture to face
    img_crop = imcrop(img, bbox);
    
    % resize images to size required by the input layer
    img_resize = imresize(img_crop, model.Layers(1).InputSize(1:2));
    % change color channel to 3 if neccessary
    if (size(img_resize,3) == 1)
        img_resize = cat(3, img_resize, img_resize, img_resize);
    end
    
    % make prediction
    [YPred, probs] = classify(model, img_resize);
    pred_prob = probs(YPred);
    if (pred_prob > threshold)
        resultID = double(string(YPred));
        threshold = pred_prob;
    end
end
end