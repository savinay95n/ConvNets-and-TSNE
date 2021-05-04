% t-SNE visualtisations for Skinny Network
clc;
clear;

dataDir= './data/wallpapers/';
checkpointDir = 'modelCheckpoints';

load('skinny_net.mat');

rng(1) % For reproducibility
Symmetry_Groups = {'P1', 'P2', 'PM' ,'PG', 'CM', 'PMM', 'PMG', 'PGG', 'CMM',...
    'P4', 'P4M', 'P4G', 'P3', 'P3M1', 'P31M', 'P6', 'P6M'};

train_folder = 'train_aug';
test_folder  = 'test_aug';

fprintf('Loading Train Filenames and Label Data...'); t = tic;
train_all = imageDatastore(fullfile(dataDir,train_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
train_all.Labels = reordercats(train_all.Labels,Symmetry_Groups);


[train, val] = splitEachLabel(train_all,.9);
fprintf('Done in %.02f seconds\n', toc(t));

fprintf('Loading Test Filenames and Label Data...'); t = tic;
test = imageDatastore(fullfile(dataDir,test_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
test.Labels = reordercats(test.Labels,Symmetry_Groups);
fprintf('Done in %.02f seconds\n', toc(t));

% Activations
train_activations = activations(net1, train, 'fc_2');
val_activations = activations(net1, val, 'fc_2');
test_activations = activations(net1, test, 'fc_2');

% tsne
Ytrain_tsne = tsne(train_activations, 'Standardize', true);
Yval_tsne = tsne(val_activations, 'Standardize', true);
Ytest_tsne = tsne(test_activations, 'Standardize', true);

% Plotting
train_plot = figure;
gscatter(Ytrain_tsne(:,1), Ytrain_tsne(:,2), train.Labels);
title('t-SNE - Train - Skinny');
saveas(train_plot, 'tsne_train_skinny.png')

val_plot = figure;
gscatter(Yval_tsne(:,1), Yval_tsne(:,2), val.Labels);
title('t-SNE - Val - Skinny');
saveas(val_plot, 'tsne_val_skinny.png')

test_plot = figure;
gscatter(Ytest_tsne(:,1), Ytest_tsne(:,2), test.Labels);
title('t-SNE - Test - Skinny');
saveas(test_plot, 'tsne_test_skinny.png')
