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
net = buildNetwork(labels, images);
net.split(.8,.2);

%* * * * * * * * * * * * * * * * * * * * * * * * * *
%    Section 2.2: Hyperparameter Initialization
%* * * * * * * * * * * * * * * * * * * * * * * * * *
% adjust network parameters
% epochs, batches, transfer function, output function, optimization
% method, etc.
fprintf('\n <<< Hyperparameter initialization >>> \n\n');
menuHyper(net);
fprintf(' << Press any key to continue >>\n');
pause();

% %* * * * * * * * * * * * * * * * * * * * * * * * * *
% %          Section 3: Train the Network
% %* * * * * * * * * * * * * * * * * * * * * * * * * *

% % trial 1
net.fit(false);

% % trial 2
% net.resetMetrics();
% net.epochs = 20;
% net.eta = .00001;
% net.batches = 1;            % BGD= 0; SGD = 1, 
% net.lambda = .1;            % L1, L2 hyperparameter
% net.mu = .4;                % momentum sensitivity hyperparameter
% net.transfer = 'leaky';       
% net.lastLayer = 'softmax';
% net.costFunction = 'cross';   
% net.optim.none = false;     % overrides all optimizations
% net.optim.lasso = false;
% net.optim.ridge = false;
% net.optim.momentum = false;
% net.optim.dropout = true;
% net.optim.early = true;
% net.fit(false);

% % trial 3
% net.resetMetrics();

% net.epochs = 20;
% net.eta = .001;
% net.batches = 64;           % BGD= 0; SGD = 1, 
% net.lambda = .1;           % L1, L2 hyperparameter
% net.mu = .7;                % momentum sensitivity hyperparameter
% net.transfer = 'leaky';       
% net.lastLayer = 'softmax';
% net.costFunction = 'mse';   
% net.optim.none = false;     % overrides all optimizations
% net.optim.lasso = false;
% net.optim.ridge = false;
% net.optim.momentum = false;
% net.optim.dropout = false;
% net.optim.early = false;
% net.fit(false);
% 

% %* * * * * * * * * * * * * * * * * * * * * * * * * *
% %        Section 4: Test
% %* * * * * * * * * * * * * * * * * * * * * * * * * *
% 

fprintf('\nLoading Project data. Please wait a moment...\n');
if ispc
    datadir = [pwd '\project\test_data.csv'];
else
    datadir = [pwd '/project/test_data.csv'];
end
data = load(datadir);
split data file into images and labels
[tstLabels, tstImages] = MNIST(data);
net.images.test = tstImages
net.images.tstLabels = tstLabels
net.images.tstEncLabels = oneHotEncoding(net.images.tstLabels);
net.predict('test')
