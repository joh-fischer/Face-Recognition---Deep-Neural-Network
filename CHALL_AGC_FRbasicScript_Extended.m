% Basic script for Face Recognition Challenge
% --------------------------------------------------------------------
% AGC Challenge  
% Universitat Pompeu Fabra
%

% Load challenge Training data
load AGC_Challenge3_Training.mat

% Provide the path to the input images, for example 
% 'C:\AGC_Challenge\images\'
imgPath = [pwd '\TRAINING\'];

% Initialize results structure
AutoRecognSTR = struct();

% Initialize timer accumulator
total_time = 0;

% Load Face Recognition model from myFaceRecognitionModel.mat
% This file must contain a single variable called 'myFRModel'
% with any information or parameter needed by the
% function 'MyFaceRecognFunction' (see below)
load myFaceRecognitionModel

s = dir('myFaceRecognitionModel.mat');
filesize = s.bytes/1e6;
fprintf(['Modelsize: ' num2str(filesize) ' MB \n']);

% Process all images in the Training set
progress = 0;
for j = 1 : length( AGC_Challenge3_TRAINING )
    A = imread( sprintf('%s%s',...
        imgPath, AGC_Challenge3_TRAINING(j).imageName ));    
    
    %try
        % Timer on
        tic;
                
        % ###############################################################
        % Your face recognition function goes here. It must accept 2 input
        % parameters:
        %
        % 1. the input image A
        % 2. the recognition model
        %
        % and it must return a single integer number as output, which can
        % be:
        % a) A number between 1 and 80 (representing one of the identities
        % in the trainig set)
        % b) A "-1" indicating that none of the 80 users is present in the
        % input image
        %
        
        autom_id = my_face_recognition_function( A, my_FRmodel );        
        % ###############################################################
        
        % Update total time
        tt = toc;
        total_time = total_time + tt;
        
    %catch
        % % If the face recognition function fails, it will be assumed that no
        % % user was detected for this input image
        % autom_id = -1;
    %end

    % Store the detection(s) in the resulst structure
    AutoRecognSTR(j).id = autom_id;
    % print progress bar
    percent = j*100/length(AGC_Challenge3_TRAINING);
    if (floor(percent/5) > progress)
        progress = floor(percent/5);
        msg = ['[' repmat('=',1,progress) repmat(' ',1,20-progress) '] - ' num2str(round(percent,2)) '%'];
        disp(msg);
    end
end
   
true_ident = [AGC_Challenge3_TRAINING.id];
pred_ident = [AutoRecognSTR.id];
% get prediction accuracy for identities
idx_ident = true_ident ~= -1;
acc_ident = sum(true_ident(idx_ident) == pred_ident(idx_ident)) / sum(idx_ident);
disp(['Accuracy identities: ' num2str( round(acc_ident*100,2) ) '%']);
% get prediction accuracy for -1
idx_unkn = true_ident == -1;
acc_unkn = sum(true_ident(idx_unkn) == pred_ident(idx_unkn)) / sum(idx_unkn);
disp(['Accuracy unknown: ' num2str( round(acc_unkn*100, 2) ) '%']);


% Compute detection score
FR_score = CHALL_AGC_ComputeRecognScores(...
    AutoRecognSTR, AGC_Challenge3_TRAINING);

% Display summary of results
fprintf(1, '\nF1-score: %.2f%% \t Total time: %dm %ds\n', ...
    100 * FR_score, int16( total_time/60),...
    int16(mod( total_time, 60)) );