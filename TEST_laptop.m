%   struct with fields:
% 
%              Name: 'Intel(R) Core(TM) i7-8750H CPU @ 2.20GHz'
%             Clock: '2201 MHz'
%             Cache: '1536 KB'
%     NumProcessors: 6
% 
% memory
% Maximum possible array:       14350 MB (1.505e+10 bytes) *
% Memory available for all arrays:       14350 MB (1.505e+10 bytes) *
% Memory used by MATLAB:        1822 MB (1.910e+09 bytes)
% Physical Memory (RAM):       16228 MB (1.702e+10 bytes)
% 
% *  Limited by System Memory (physical + swap file) available.
clc;
%clear;
dd = firstMenu();
data = load(dd);
[labels, images] = MNIST(data);
net = menuNetwork(labels, images);
net.split(.7,.2,.1);

% directly set hyperparameters
net.epochs = 500;
net.eta = .00001;
net.batches = 1;           % BGD= 0; SGD = 1, 
net.lambda = .1;           % L1, L2 hyperparameter
net.mu = .4;                % momentum sensitivity hyperparameter
net.transfer = 'leaky';       
net.lastLayer = 'softmax';
net.costFunction = 'cross';  
net.optim.none = false;     % overrides all optimizations
net.optim.lasso = false;
net.optim.ridge = true;
net.optim.momentum = true;
net.optim.dropout = false;
net.droprate = .85;
net.optim.early = false;

net.fit(false);
net.predict('test');
