% Code to generate 128X128 images from 256X256 images in the original data
% set. Run the code once and save the data.
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



% Split with validation set
[train, val] = splitEachLabel(train_all,.9);
fprintf('Done in %.02f seconds\n', toc(t));

fprintf('Loading Test Filenames and Label Data...'); t = tic;
test = imageDatastore(fullfile(dataDir,test_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
test.Labels = reordercats(test.Labels,Symmetry_Groups);
fprintf('Done in %.02f seconds\n', toc(t));

%% Data Generation - 256X256 to 128X128 images
% Run it only once and save data
mkdir('./data/wallpapers/train_original_transfer');
mkdir('./data/wallpapers/test_original_transfer');
train_folder = 'train_original_transfer';
test_folder  = 'test_original_transfer';
for i = 1:length(train_all.Files)
    I_tr = imresize(imread(train_all.Files{i}),[128,128]);
    I_tst = imresize(imread(test.Files{i}),[128,128]);

    C_tr = strsplit(train_all.Files{i},'\');
    C_tst = strsplit(test.Files{i},'\');
    filename_tr = strcat('./data/wallpapers/train_original_transfer/',C_tr{end});
    filename_tst = strcat('./data/wallpapers/test_original_transfer/',C_tst{end});
    imwrite(I_tr,filename_tr);
    imwrite(I_tst,filename_tst);

end

train_all_transfer = imageDatastore(fullfile(dataDir,train_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
train_all_transfer.Labels = reordercats(train_all.Labels,Symmetry_Groups);

test_transfer = imageDatastore(fullfile(dataDir,test_folder),'IncludeSubfolders',true,'LabelSource',...
    'foldernames');
test_transfer.Labels = reordercats(test.Labels,Symmetry_Groups);

[train, val] = splitEachLabel(train_all_transfer,.9);