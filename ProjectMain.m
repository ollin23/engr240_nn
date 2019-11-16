% This is the main script file for the project
clear;clc; 

% setup formatting
format compact;
format shortg;

fprintf('\nLoading Project training data. Please wait a moment...\n');

%* * * * * * * * * * * * * * * * * * * * * * * * * *
%          Section 1: Data Preparation
%* * * * * * * * * * * * * * * * * * * * * * * * * *
datadir = firstMenu();

% load data
data = load(datadir);
 
% split data file into images and labels
[labels, images] = MNIST(data);

%* * * * * * * * * * * * * * * * * * * * * * * * * *
%        Section 2.1: Parameter Initialization
%* * * * * * * * * * * * * * * * * * * * * * * * * *
% create the network
net = menuNetwork(images, labels);

%* * * * * * * * * * * * * * * * * * * * * * * * * *
%    Section 2.2: Hyperparameter Initialization
%* * * * * * * * * * * * * * * * * * * * * * * * * *
% adjust network parameters
% epochs, batches, transfer function, output function, optimization
% method, and 

fprintf('\n <<< Hyperparameter initialization >>> \n\n');
% set defaults
% * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
% REMOVE THIS SECTION FROM FINAL VERSION
% * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
% trial 1
net.epochs = 500;
net.eta = .001;
net.batches = 64;           % BGD= 0; SGD = 1, 
net.lambda = .01;           % L1, L2 hyperparameter
net.mu = .7;                % momentum sensitivity hyperparameter
net.transfer = 'leaky';       
net.lastLayer = 'softmax';
net.costFunction = 'mse';   
net.optim.none = false;     % overrides all optimizations
net.optim.lasso = false;
net.optim.ridge = false;
net.optim.momentum = false;
net.optim.dropout = false;
net.optim.early = true;
net.fit();

% trial 1.5
net.reset();
net.epochs = 500;
net.eta = .001;
net.batches = 64;           % BGD= 0; SGD = 1, 
net.lambda = .01;           % L1, L2 hyperparameter
net.mu = .7;                % momentum sensitivity hyperparameter
net.transfer = 'leaky';       
net.lastLayer = 'softmax';
net.costFunction = 'mse';   
net.optim.none = false;     % overrides all optimizations
net.optim.lasso = false;
net.optim.ridge = false;
net.optim.momentum = false;
net.optim.dropout = false;
net.optim.early = false;
net.fit();

% give user opportunity to change defaults
% menuHyper(net);
% fprintf(' << Press any key to continue >>\n');
% pause();

%* * * * * * * * * * * * * * * * * * * * * * * * * *
%          Section 3: Train the Network
%* * * * * * * * * * * * * * * * * * * * * * * * * *
% train the network
%net.fit();

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
% net.predict()

% trial 2
net.reset();
net.trial = net.trial + 1;
net.epochs = 500;
net.eta = .001;
net.batches = 0;            % BGD= 0; SGD = 1, 
net.lambda = .01;           % L1, L2 hyperparameter
net.mu = .7;                % momentum sensitivity hyperparameter
net.transfer = 'leaky';       
net.lastLayer = 'softmax';
net.costFunction = 'mse';   % 'cross' pairs with softmax
net.optim.none = true;      % 'true' not very useful with tanh and softmax
net.optim.lasso = false;
net.optim.ridge = false;
net.optim.momentum = false;
net.optim.dropout = false;
net.optim.early = false;

net.fit();

% % trial 3
% net.reset();
% net.trial = net.trial +1;
% net.epochs = 1000;
% net.eta = .001;
% net.batches = 1;              % BGD= 0; SGD = 1, 
% net.lambda = .01;             % L1, L2 hyperparameter
% net.mu = .7;                  % momentum sensitivity hyperparameter
% net.transfer = 'leaky';       
% net.lastLayer = 'softmax';
% net.costFunction = 'mse';     % 'cross' pairs with softmax
% net.optim.none = true;        % 'true' not very useful with tanh and softmax
% net.optim.lasso = false;
% net.optim.ridge = false;
% net.optim.momentum = false;
% net.optim.dropout = false;
% net.optim.early = false;
% 
% net.fit();
% 
% 
% % trial 4
% net.reset();
% net.trial = net.trial + 1;
% net.epochs = 1000;
% net.eta = .001;
% net.batches = 64;             % BGD= 0; SGD = 1, 
% net.lambda = .01;             % L1, L2 hyperparameter
% net.mu = .7;                  % momentum sensitivity hyperparameter
% net.transfer = 'leaky';       
% net.lastLayer = 'softmax';
% net.costFunction = 'mse';     % 'cross' pairs with softmax
% net.optim.none = true;        % 'true' not very useful with tanh and softmax
% net.optim.lasso = false;
% net.optim.ridge = false;
% net.optim.momentum = false;
% net.optim.dropout = false;
% net.optim.early = false;
% 
% net.fit();
% 
%
% % trial 5
% net.reset();
% net.trial = net.trial + 1;
% net.epochs = 1000;
% net.eta = .001;
% net.batches = 64;             % BGD= 0; SGD = 1, 
% net.lambda = .01;             % L1, L2 hyperparameter
% net.mu = .7;                  % momentum sensitivity hyperparameter
% net.transfer = 'leaky';       
% net.lastLayer = 'softmax';
% net.costFunction = 'mse';     % 'cross' pairs with softmax
% net.optim.none = false;       % 'true' not very useful with tanh and softmax
% net.optim.lasso = true;
% net.optim.ridge = false;
% net.optim.momentum = false;
% net.optim.dropout = false;
% net.optim.early = false;
% 
% net.fit();
% 
%
% % trial 6
% net.reset();
% net.trial = net.trial + 1;
% net.epochs = 1000;
% net.eta = .001;
% net.batches = 64;             % BGD= 0; SGD = 1, 
% net.lambda = .01;             % L1, L2 hyperparameter
% net.mu = .7;                  % momentum sensitivity hyperparameter
% net.transfer = 'leaky';       
% net.lastLayer = 'softmax';
% net.costFunction = 'mse';     % 'cross' pairs with softmax
% net.optim.none = false;       % 'true' not very useful with tanh and softmax
% net.optim.lasso = true;
% net.optim.ridge = true;
% net.optim.momentum = false;
% net.optim.dropout = false;
% net.optim.early = false;
% 
% net.fit();
% 
% 
% % trial 7
% net.reset();
% net.trial = net.trial + 1;
% net.epochs = 1000;
% net.eta = .001;
% net.batches = 64;             % BGD= 0; SGD = 1, 
% net.lambda = .01;             % L1, L2 hyperparameter
% net.mu = .7;                  % momentum sensitivity hyperparameter
% net.transfer = 'leaky';       
% net.lastLayer = 'softmax';
% net.costFunction = 'mse';     % 'cross' pairs with softmax
% net.optim.none = false;       % 'true' not very useful with tanh and softmax
% net.optim.lasso = true;
% net.optim.ridge = true;
% net.optim.momentum = true;
% net.optim.dropout = false;
% net.optim.early = false;
% 
% net.fit();
% 
%
% % trial 8
% net.reset();
% net.trial = net.trial + 1;
% net.epochs = 1500;
% net.eta = .001;
% net.batches = 64;             % BGD= 0; SGD = 1, 
% net.lambda = .01;             % L1, L2 hyperparameter
% net.mu = .7;                  % momentum sensitivity hyperparameter
% net.transfer = 'leaky';       
% net.lastLayer = 'softmax';
% net.costFunction = 'mse';     % 'cross' pairs with softmax
% net.optim.none = false;       % 'true' not very useful with tanh and softmax
% net.optim.lasso = true;
% net.optim.ridge = true;
% net.optim.momentum = true;
% net.optim.dropout = false;
% net.droprate = .9;
% net.optim.early = false;
% 
% net.fit();
% 
%
% % trial 9
% net.reset();
% net.trial = net.trial + 1;
% net.epochs = 1500;
% net.eta = .001;
% net.batches = 64;             % BGD= 0; SGD = 1, 
% net.lambda = .01;             % L1, L2 hyperparameter
% net.mu = .7;                  % momentum sensitivity hyperparameter
% net.transfer = 'leaky';       
% net.lastLayer = 'softmax';
% net.costFunction = 'mse';     % 'cross' pairs with softmax
% net.optim.none = false;       % 'true' not very useful with tanh and softmax
% net.optim.lasso = true;
% net.optim.ridge = true;
% net.optim.momentum = true;
% net.optim.dropout = true;
% net.droprate = .9;
% net.optim.early = false;
% 
% net.fit();
% 
%
% % trial 10
% net.reset();
% net.trial = net.trial + 1;
% net.epochs = 1500;
% net.eta = .001;
% net.batches = 64;             % BGD= 0; SGD = 1, 
% net.lambda = .01;             % L1, L2 hyperparameter
% net.mu = .7;                  % momentum sensitivity hyperparameter
% net.transfer = 'leaky';       
% net.lastLayer = 'softmax';
% net.costFunction = 'mse';     % 'cross' pairs with softmax
% net.optim.none = false;       % 'true' not very useful with tanh and softmax
% net.optim.lasso = true;
% net.optim.ridge = true;
% net.optim.momentum = true;
% net.optim.dropout = true;
% net.droprate = .5;
% net.optim.early = false;
% 
% net.fit();