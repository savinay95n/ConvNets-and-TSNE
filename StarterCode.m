%*************************************************************************
% Starter Code - for 5 and 10 epochs and learning rates of 5e^-4 and 1e^-4
%*************************************************************************
clc;
clear;

dataDir= './data/wallpapers/';
checkpointDir = 'modelCheckpoints';

rng(1) % For reproducibility
Symmetry_Groups = {'P1', 'P2', 'PM' ,'PG', 'CM', 'PMM', 'PMG', 'PGG', 'CMM',...
    'P4', 'P4M', 'P4G', 'P3', 'P3M1', 'P31M', 'P6', 'P6M'};

train_folder = 'train';
test_folder  = 'test';
% uncomment after you create the augmentation dataset
% train_folder = 'train_aug';
% test_folder  = 'test_aug';
fprintf('Loading Train Filenames and Label Data...'); t = tic;
train_all = imageDatastore(fullfile(dataDir,train_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
train_all.Labels = reordercats(train_all.Labels,Symmetry_Groups);
%%
% Split with validation set
[train, val] = splitEachLabel(train_all,.9);
fprintf('Done in %.02f seconds\n', toc(t));

fprintf('Loading Test Filenames and Label Data...'); t = tic;
test = imageDatastore(fullfile(dataDir,test_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
test.Labels = reordercats(test.Labels,Symmetry_Groups);
fprintf('Done in %.02f seconds\n', toc(t));

%%
rng('default');
numEpochs = 10; % 5 for both learning rates
batchSize = 250;
nTraining = length(train.Labels);

% Define the Network Structure, To add more layers, copy and paste the
% lines such as the example at the bottom of the code
%  CONV -> ReLU -> POOL -> FC -> DROPOUT -> FC -> SOFTMAX 
layers = [
    imageInputLayer([256 256 1]); % Input to the network is a 256x256x1 sized image 
    convolution2dLayer(5,20,'Padding',[2 2],'Stride', [2,2]);  % convolution layer with 20, 5x5 filters
    reluLayer();  % ReLU layer
    maxPooling2dLayer(2,'Stride',2); % Max pooling layer
    fullyConnectedLayer(25); % Fullly connected layer with 50 activations
    dropoutLayer(.25); % Dropout layer
    fullyConnectedLayer(17); % Fully connected with 17 layers
    softmaxLayer(); % Softmax normalization layer
    classificationLayer(); % Classification layer
    ];

if ~exist(checkpointDir,'dir'); mkdir(checkpointDir); end
% Set the training options
options = trainingOptions('sgdm','MaxEpochs',20,... 
    'InitialLearnRate',5e-4,...% learning rate
    'CheckpointPath', checkpointDir,...
    'MiniBatchSize', batchSize, ...
    'MaxEpochs',numEpochs);
    % uncommand and add the line below to the options above if you have 
    % version 17a or above to see the learning in realtime
    %'OutputFcn',@plotTrainingAccuracy,... 

% Train the network, info contains information about the training accuracy
% and loss
 t = tic;
[net1,info1] = trainNetwork(train,layers,options);
fprintf('Trained in in %.02f seconds\n', toc(t));

error_plot = figure;
plotTrainingAccuracy_All(info1,numEpochs);
saveas(error_plot, ['Error plot for epoch ', num2str(numEpochs), '.png']);

%%
% Test on the training data
YTrain_pred = classify(net1,train);
train_acc = mean(YTrain_pred==train.Labels);

train_con_mat = confusionmat(sort(grp2idx(train.Labels)), sort(grp2idx(YTrain_pred)));
train_class_mat = train_con_mat./(meshgrid(countcats(train.Labels))');

filename = ['Train_Confusion_Mat_', num2str(numEpochs), '.xlsx'];
xlswrite(filename, train_con_mat,'Sheet1','A1');

filename = ['Train_Classification_Mat_', num2str(numEpochs), '.xlsx'];
xlswrite(filename, train_class_mat,'Sheet1','A1');

%%

% Test on the validation data
YVal_pred = classify(net1,val);
val_acc = mean(YVal_pred==val.Labels);

val_con_mat = confusionmat(sort(grp2idx(val.Labels)), sort(grp2idx(YVal_pred)));
val_class_mat = val_con_mat./(meshgrid(countcats(val.Labels))');

filename = ['Val_Confusion_Mat_', num2str(numEpochs), '.xlsx'];
xlswrite(filename, val_con_mat,'Sheet1','A1');

filename = ['Val_Classification_Mat_', num2str(numEpochs), '.xlsx'];
xlswrite(filename, val_class_mat,'Sheet1','A1');

% Test on the Test data
YTest_pred = classify(net1,test);
test_acc = mean(YTest_pred==test.Labels);

test_con_mat = confusionmat(sort(grp2idx(test.Labels)), sort(grp2idx(YTest_pred)));
test_class_mat = test_con_mat./(meshgrid(countcats(test.Labels))');

filename = ['Test_Confusion_Mat_', num2str(numEpochs), '.xlsx'];
xlswrite(filename, test_con_mat,'Sheet1','A1');

filename = ['Test_Classification_Mat_', num2str(numEpochs), '.xlsx'];
xlswrite(filename, test_class_mat,'Sheet1','A1');


%%
% It seems like it isn't converging after looking at the graph but lets
%   try dropping the learning rate to show you how.  

rng('default');
numEpochs = 10; % 5 for both learning rates
batchSize = 250;
nTraining = length(train.Labels);
InitialLearnRate = 1e-4;
options = trainingOptions('sgdm','MaxEpochs',20,...
    'InitialLearnRate',1e-4,... % learning rate
    'CheckpointPath', checkpointDir,...
    'MiniBatchSize', batchSize, ...
    'MaxEpochs',numEpochs);
    % uncommand and add the line below to the options above if you have 
    % version 17a or above to see the learning in realtime
%     'OutputFcn',@plotTrainingAccuracy,...

 t = tic;
[net2,info2] = trainNetwork(train,net1.Layers,options);
fprintf('Trained in in %.02f seconds\n', toc(t));

error_plot = figure;
plotTrainingAccuracy_All(info2,numEpochs);
saveas(error_plot, ['Error plot for epoch ', num2str(numEpochs), '_', num2str(InitialLearnRate), '.png']);

%%
% Test on the training data
YTrain_pred = classify(net2,train);
train_acc = mean(YTrain_pred==train.Labels);

train_con_mat = confusionmat(sort(grp2idx(train.Labels)), sort(grp2idx(YTrain_pred)));
train_class_mat = train_con_mat./(meshgrid(countcats(train.Labels))');

filename = ['Train_Confusion_Mat_', num2str(numEpochs), '_', num2str(InitialLearnRate),'.xlsx'];
xlswrite(filename, train_con_mat,'Sheet1','A1');

filename = ['Train_Classification_Mat_', num2str(numEpochs), '_', num2str(InitialLearnRate), '.xlsx'];
xlswrite(filename, train_class_mat,'Sheet1','A1');

%%
% Test on the validation data
YVal_pred = classify(net2,val);
val_acc = mean(YVal_pred==val.Labels);

val_con_mat = confusionmat(sort(grp2idx(val.Labels)), sort(grp2idx(YVal_pred)));
val_class_mat = val_con_mat./(meshgrid(countcats(val.Labels))');

filename = ['Val_Confusion_Mat_', num2str(numEpochs), '_', num2str(InitialLearnRate), '.xlsx'];
xlswrite(filename, val_con_mat,'Sheet1','A1');

filename = ['Val_Classification_Mat_', num2str(numEpochs), '_', num2str(InitialLearnRate), '.xlsx'];
xlswrite(filename, val_class_mat,'Sheet1','A1');

% Test on the Test data
YTest_pred = classify(net2,test);
test_acc = mean(YTest_pred==test.Labels);

test_con_mat = confusionmat(sort(grp2idx(test.Labels)), sort(grp2idx(YTest_pred)));
test_class_mat = test_con_mat./(meshgrid(countcats(test.Labels))');

filename = ['Test_Confusion_Mat_', num2str(numEpochs), '_', num2str(InitialLearnRate), '.xlsx'];
xlswrite(filename, test_con_mat,'Sheet1','A1');

filename = ['Test_Classification_Mat_', num2str(numEpochs), '_', num2str(InitialLearnRate), '.xlsx'];
xlswrite(filename, test_class_mat,'Sheet1','A1');


% It seems like continued training would improve the scores

