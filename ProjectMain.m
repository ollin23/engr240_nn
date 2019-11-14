% This is the main script file for the project
clear;clc; 

% clear layers nodes outputCount response counter;
format compact;

fprintf('\nLoading Project training data. Please wait a moment...\n');

%* * * * * * * * * * * * * * * * * * * * * * * * * *
%          Section 1: Data Preparation
%* * * * * * * * * * * * * * * * * * * * * * * * * *
% % actual data
% datadir = [pwd '\project\train_data.csv'];

% the below is test data of 1000 samples to keep training times short
% during build
datadir = [pwd '\project\samples1k.csv'];

% load data
data = load(datadir);
 
% split data file into images and labels
[labels, images] = MNIST(data);



%* * * * * * * * * * * * * * * * * * * * * * * * * *
%        Section 2.1: Parameter Initialization
%* * * * * * * * * * * * * * * * * * * * * * * * * *
% get size of output vector
[sampleCount, imgSize] = size(images);
nodes = imgSize;
outputCount = length(unique(labels));

%determine default count of nodes in layer and number of layers
counter = 1;
layers(counter) = nodes;
while (2*outputCount <= nodes)
    nodes = round(sqrt(layers(counter))) + outputCount;
    counter = counter + 1;
    layers(counter) = nodes;
end
layers = [layers outputCount];

% get input from user for number of layers and the number of nodes in each
% layer
fprintf('Do you want to use the default design? (Y/n)  ');
answer = input('','s');
answer = lower(answer);
if length(answer) < 1
    answer = 'y';
end
if (strcmp(answer,'n') || strcmp(answer(1),'n'))
    nodesEachLayer = menu();
    layers = [imgSize nodesEachLayer outputCount];
end

% create network
net = Network(layers);

% display the network topology
displayNetworkDesign(layers);
disp('<< Press Enter to continue...>>');
% pause();

% encode the labels
% add network metric parameters
net.encodedLabels = oneHotEncoding(labels);
net.images = images;
net.labels = labels;

%* * * * * * * * * * * * * * * * * * * * * * * * * *
%    Section 2.2: Hyperparameter Initialization
%* * * * * * * * * * * * * * * * * * * * * * * * * *
% adjust network parameters
% epochs, batches, transfer function, output function, optimization
% method, and 

fprintf('\n ((hyperparameter initialization))\n\n');
% set defaults
% * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
% REMOVE THIS SECTION FROM FINAL VERSION
% * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
net.epochs = 1000;
net.eta = .0001;
net.batches = 1;   % SGD (stochastic gradient descent) = sampleSize, 
net.lambda = .01;
net.mu = .7;
net.transfer = 'leaky';       % default tanh hyperbolic tangent
net.lastLayer = 'softmax';
net.optim.none = true;   % not particularly useful with tanh and softmax
net.optim.lasso = false;
net.optim.ridge = false;
net.optim.momentum = false;
net.optim.dropout = false;
net.costFunction = 'mse';  % pairs with softmax

% give user opportunity to change defaults
menuHyper(net);
fprintf(' << Press any key to continue >>\n');
pause();


%* * * * * * * * * * * * * * * * * * * * * * * * * *
%          Section 3: Train the Network
%* * * * * * * * * * * * * * * * * * * * * * * * * *
% train the network
% shuffledIndex = randperm(length(nn.images));
% nn.images= nn.images(shuffledIndex,:);
fit(net);

%* * * * * * * * * * * * * * * * * * * * * * * * * *
%        Section 4: Test
%* * * * * * * * * * * * * * * * * * * * * * * * * *

% data = load('test_data.csv');
% fprintf('\nLoading Project data. Please wait a moment...\n');
% datadir = [pwd '\project\test_data.csv'];
% data = load(datadir);
% split data file into images and labels
% [tstLabels, tstImages] = MNIST(data);
% net.images = tstImages
% net.labels = tstLabels


