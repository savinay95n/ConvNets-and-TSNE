% Code for Skinny Network on Original Dataset
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
%train_folder = 'train_aug';
%test_folder  = 'test_aug';
fprintf('Loading Train Filenames and Label Data...'); t = tic;
train_all = imageDatastore(fullfile(dataDir,train_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
train_all.Labels = reordercats(train_all.Labels,Symmetry_Groups);
test = imageDatastore(fullfile(dataDir,test_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
test.Labels = reordercats(test.Labels,Symmetry_Groups);

train_folder = 'train_original_transfer';
test_folder  = 'test_original_transfer';
train_transfer = imageDatastore(fullfile(dataDir,train_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
train_transfer.Labels = reordercats(train_all.Labels,Symmetry_Groups);
test_transfer = imageDatastore(fullfile(dataDir,test_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
test_transfer.Labels = reordercats(test.Labels,Symmetry_Groups);
[train, val] = splitEachLabel(train_transfer,.9);




%%
rng('default');
numEpochs = 5; % 5 for both learning rates
batchSize = 250;
nTraining = length(train.Labels);



% Set the training options
options = trainingOptions('sgdm','MaxEpochs',25,... 
    'InitialLearnRate',0.5e-3,...% learning rate
    'CheckpointPath', checkpointDir,...
    'MiniBatchSize', batchSize, ...
    'MaxEpochs',numEpochs);
    % uncommand and add the line below to the options above if you have 
    % version 17a or above to see the learning in realtime
    %'OutputFcn',@plotTrainingAccuracy,... 

% Train the network, info contains information about the training accuracy
% and loss
t = tic;
% Loading trained Model
load('skinny_net.mat');
[netT,infoT] = trainNetwork(train,net1.Layers,options);
fprintf('Trained in in %.02f seconds\n', toc(t));

error_plot = figure;
plotTrainingAccuracy_All(infoT,numEpochs);
saveas(error_plot, ['Skinny_Big_Error plot for epoch 5_transfer', num2str(numEpochs), '.png']);


% Test on the training data
YTrain_Pred = classify(netT,train);
train_acc = mean(YTrain_Pred==train.Labels);

train_con_mat = confusionmat(sort(grp2idx(train.Labels)), sort(grp2idx(YTrain_Pred)));
train_class_mat = train_con_mat./(meshgrid(countcats(train.Labels))');

filename = ['Skinny_Big_Train_Confusion_Mat 5_transfer', num2str(numEpochs), '.xlsx'];
xlswrite(filename, train_con_mat,'Sheet1','A1');

filename = ['Skinny_Big_Train_Classification_Mat 5_transfer', num2str(numEpochs), '.xlsx'];
xlswrite(filename, train_class_mat,'Sheet1','A1');



% Test on the validation data
YVal_Pred = classify(netT,val);
val_acc = mean(YVal_Pred==val.Labels);

val_con_mat = confusionmat(sort(grp2idx(val.Labels)), sort(grp2idx(YVal_Pred)));
val_class_mat = val_con_mat./(meshgrid(countcats(val.Labels))');

filename = ['Skinny_Big_Val_Confusion_Mat 5_transfer', num2str(numEpochs), '.xlsx'];
xlswrite(filename, val_con_mat,'Sheet1','A1');

filename = ['Skinny_Big_Val_Classification_Mat 5_transfer', num2str(numEpochs), '.xlsx'];
xlswrite(filename, val_class_mat,'Sheet1','A1');

% Test on the Test data
YTest_Pred = classify(netT,test_transfer);
test_acc = mean(YTest_Pred==test_transfer.Labels);

test_con_mat = confusionmat(sort(grp2idx(test_transfer.Labels)), sort(grp2idx(YTest_Pred)));
test_class_mat = test_con_mat./(meshgrid(countcats(test_transfer.Labels))');

filename = ['Skinny_Big_Test_Confusion_Mat 5_transfer', num2str(numEpochs), '.xlsx'];
xlswrite(filename, test_con_mat,'Sheet1','A1');

filename = ['Skinny_Big_Test_Classification_Mat 5_transfer', num2str(numEpochs), '.xlsx'];
xlswrite(filename, test_class_mat,'Sheet1','A1');
