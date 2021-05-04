% Skinny Network Code
clc;
clear;

dataDir= './data/wallpapers/';
checkpointDir = 'modelCheckpoints';

rng(1) % For reproducibility
Symmetry_Groups = {'P1', 'P2', 'PM' ,'PG', 'CM', 'PMM', 'PMG', 'PGG', 'CMM',...
    'P4', 'P4M', 'P4G', 'P3', 'P3M1', 'P31M', 'P6', 'P6M'};

%train_folder = 'train';
%test_folder  = 'test';
% uncomment after you create the augmentation dataset
train_folder = 'train_aug';
test_folder  = 'test_aug';
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
numEpochs = 15; % 5 for both learning rates
batchSize = 250;
nTraining = length(train.Labels);

% Define the Network Structure, To add more layers, copy and paste the
% lines such as the example at the bottom of the code
%  CONV -> ReLU -> POOL -> FC -> DROPOUT -> FC -> SOFTMAX 
layers = [
    imageInputLayer([128 128 1]); % Input to the network is a 256x256x1 sized image 
    convolution2dLayer(5,40,'Padding',[2 2],'Stride', [1,1]);  % convolution layer with 20, 5x5 filters
    reluLayer();  % ReLU layer
    maxPooling2dLayer(2,'Stride',2); % Max pooling layer
    convolution2dLayer(5,40,'Padding',[1 1],'Stride', [1,1]);  % convolution layer with 20, 5x5 filters
    reluLayer();  % ReLU layer
    maxPooling2dLayer(2,'Stride',2); % Max pooling layer
    convolution2dLayer(3,80,'Padding',[1 1],'Stride', [1,1]);  % convolution layer with 20, 5x5 filters
    reluLayer();  % ReLU layer
    maxPooling2dLayer(2,'Stride',2); % Max pooling layer
    fullyConnectedLayer(100); % Fully connected with 17 layers
    dropoutLayer(0.25);
    fullyConnectedLayer(17); % Fully connected with 17 layers
    softmaxLayer(); % Softmax normalization layer
    classificationLayer(); % Classification layer
    ];

if ~exist(checkpointDir,'dir'); mkdir(checkpointDir); end


% Set the training options
options = trainingOptions('sgdm','MaxEpochs',25,... 
    'InitialLearnRate',1e-3,...% learning rate
    'CheckpointPath', checkpointDir,...
    'MiniBatchSize', batchSize, ...
    'MaxEpochs',numEpochs);
    % uncommand and add the line below to the options above if you have 
    % version 17a or above to see the learning in realtime
    %'OutputFcn',@plotTrainingAccuracy,... 

% Train the network, info contains information about the training accuracy
% and loss
 t = tic;
[net1,info1] = trainNetwork(train,net.Layers,options);
fprintf('Trained in in %.02f seconds\n', toc(t));

error_plot = figure;
plotTrainingAccuracy_All(info1,numEpochs);
saveas(error_plot, ['Skinny_Big_Error plot for epoch 5_2', num2str(numEpochs), '.png']);


% Test on the training data
YTrain_Pred = classify(net1,train);
train_acc = mean(YTrain_Pred==train.Labels);

train_con_mat = confusionmat(sort(grp2idx(train.Labels)), sort(grp2idx(YTrain_Pred)));
train_class_mat = train_con_mat./(meshgrid(countcats(train.Labels))');

filename = ['Skinny_Big_Train_Confusion_Mat 5_2', num2str(numEpochs), '.xlsx'];
xlswrite(filename, train_con_mat,'Sheet1','A1');

filename = ['Skinny_Big_Train_Classification_Mat 5_2', num2str(numEpochs), '.xlsx'];
xlswrite(filename, train_class_mat,'Sheet1','A1');



% Test on the validation data
YVal_Pred = classify(net1,val);
val_acc = mean(YVal_Pred==val.Labels);

val_con_mat = confusionmat(sort(grp2idx(val.Labels)), sort(grp2idx(YVal_Pred)));
val_class_mat = val_con_mat./(meshgrid(countcats(val.Labels))');

filename = ['Skinny_Big_Val_Confusion_Mat 5_2', num2str(numEpochs), '.xlsx'];
xlswrite(filename, val_con_mat,'Sheet1','A1');

filename = ['Skinny_Big_Val_Classification_Mat 5_2', num2str(numEpochs), '.xlsx'];
xlswrite(filename, val_class_mat,'Sheet1','A1');

% Test on the Test data
YTest_Pred = classify(net1,test);
test_acc = mean(YTest_Pred==test.Labels);

test_con_mat = confusionmat(sort(grp2idx(test.Labels)), sort(grp2idx(YTest_Pred)));
test_class_mat = test_con_mat./(meshgrid(countcats(test.Labels))');

filename = ['Skinny_Big_Test_Confusion_Mat 5_2', num2str(numEpochs), '.xlsx'];
xlswrite(filename, test_con_mat,'Sheet1','A1');

filename = ['Skinny_Big_Test_Classification_Mat 5_2', num2str(numEpochs), '.xlsx'];
xlswrite(filename, test_class_mat,'Sheet1','A1');


