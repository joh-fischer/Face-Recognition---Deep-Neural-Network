% Face Recognition Model
% @authors:     Johannes S. Fischer, Oriol Guardia
% @course:      Face and Gesture Analysis

clear;

trainingImgPath = 'TRAINING_DNN';

% General Settings
Settings.layersToFreeze = 16;
Settings.plot           = false;
Settings.saveModel      = true;
Settings.trainRatio     = 0.8;
% Learning Settings
Settings.miniBatchSize  = 64;
Settings.solverName     = 'adam';       % 'sgdm' | 'rmsprop' | 'adam'
Settings.maxEpochs      = 4;
Settings.initLearnRate  = 3e-4;
Settings.verbose        = false;
Settings.ExecEnvironm   = 'gpu';

% load pretrained network
net = resnet18;
modelname = 'resnet18';

%% Preparation
% get image input size
img_size = net.Layers(1).InputSize;

% load training dataset
imds = imageDatastore(trainingImgPath, ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames'); 
[imdsTrain, imds_val_test] = splitEachLabel(imds, Settings.trainRatio);
[imdsValidation, imdsTest] = splitEachLabel(imds_val_test, 0.75);

disp(['Training Files: ' num2str(length(imdsTrain.Files)) ]);
disp(['Validation Files: ' num2str(length(imdsValidation.Files)) ]);
disp(['Test Files: ' num2str(length(imdsTest.Files)) ]);

% add folders with support functions to path
addpath([matlabroot '/examples/nnet/main/']);

% to retrain the network to fit our data we need to replace the last
% learnable layer and the classification layer
% manually by lgraph.Layers(end), lgraph.Layers(end-2) [end-1 is softmax]
lgraph = layerGraph(net);
[learnableLayer, classLayer] = findLayersToReplace(lgraph);

% get number of classes
numClasses = numel(categories(imdsTrain.Labels));

% replace last learnable layer (either with a fully connected layer or a
% convolution 2D layer) with 80 nodes (one per class)
if isa(learnableLayer,'nnet.cnn.layer.FullyConnectedLayer')
    newLearnableLayer = fullyConnectedLayer(numClasses, ...
        'Name','new_fc', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
elseif isa(learnableLayer,'nnet.cnn.layer.Convolution2DLayer')
    newLearnableLayer = convolution2dLayer(1,numClasses, ...
        'Name','new_conv', ...
        'WeightLearnRateFactor',10, ...
        'BiasLearnRateFactor',10);
end
lgraph = replaceLayer(lgraph, learnableLayer.Name, newLearnableLayer);

% replace classification layer
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph, classLayer.Name, newClassLayer);

% plot the new layers in the graph
if Settings.plot
    figure('Units','normalized','Position',[0.3 0.3 0.4 0.4]);
    plot(lgraph)
    ylim([0,10])
end

% freeze the weights of the first layers to prevent overfitting
if (Settings.layersToFreeze)
layers = lgraph.Layers;
connections = lgraph.Connections;
layers(1:Settings.layersToFreeze) = freezeWeights( layers(1:Settings.layersToFreeze) );
lgraph = createLgraphUsingConnections( layers, connections );
end

% preprocess images, s.t. size & color fits
% also prevent overfitting by rotating images
pixelRange = [-30 30];
scaleRange = [0.9 1.1];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation', pixelRange, ...
    'RandYTranslation', pixelRange, ...
    'RandXScale', scaleRange, ...
    'RandYScale', scaleRange);
augimdsTrain = augmentedImageDatastore(img_size(1:2),imdsTrain, ...
    'DataAugmentation', imageAugmenter,...
    'ColorPreprocessing', 'gray2rgb');

% validation images without augmentation
augimdsValidation = augmentedImageDatastore(img_size(1:2), imdsValidation,...
    'ColorPreprocessing', 'gray2rgb');

valFrequency = floor( numel(augimdsTrain.Files) / Settings.miniBatchSize);
options = trainingOptions(Settings.solverName, ...
    'MiniBatchSize',Settings.miniBatchSize, ...
    'MaxEpochs', Settings.maxEpochs, ...
    'InitialLearnRate', Settings.initLearnRate, ...
    'Shuffle','every-epoch', ...
    'ValidationData', augimdsValidation, ...
    'ValidationFrequency', valFrequency, ...
    'Verbose',false, ...
    'Plots','training-progress', ...
    'ExecutionEnvironment', Settings.ExecEnvironm);


%% Train network
tic
disp('Start Training');
net = trainNetwork(augimdsTrain, lgraph, options);
toc

%% test network
% test images without preprocessing
augimdsTest = augmentedImageDatastore(img_size(1:2), imdsTest,...
    'ColorPreprocessing', 'gray2rgb');
[YPred, probs] = classify(net, augimdsTest);
accuracy = mean(YPred == imdsTest.Labels);
disp([ 'Accuracy: ' num2str(accuracy) ]);

% display some random images and their labels (with probabilities)
idx = randperm(numel(imdsTest.Files), 16);
figure
for i = 1:16
    subplot(4,4,i)
    I = readimage(imdsTest,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title("P: " + string(label) + " | T: " + string(imdsTest.Labels(idx(i))) + " | " + num2str(100*max(probs(idx(i),:)),3) + "%");
end

%% Save model
if (Settings.saveModel)
   Settings.accuracy = accuracy;
   date_str = datestr(now,'mmdd-HHMM');
   filename = [ date_str '_' modelname '_acc' num2str( round(accuracy*100,0) ) '_DNN-model.mat'];
   save([pwd '/models/' filename], 'net', 'Settings', 'accuracy');
end