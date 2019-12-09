clc;
clear;
dd = firstMenu();
data = load(dd);
[labels, images] = MNIST(data);
net = menuNetwork(labels, images);
net.split(.8,.2,);

% directly set hyperparameters
net.epochs = 25;
net.eta = 1e-6;
net.batches = 1;           % BGD= 0; SGD = 1, 
net.lambda = .1;           % L1, L2 hyperparameter
net.mu = .4;                % momentum sensitivity hyperparameter
net.transfer = 'leaky';       
net.lastLayer = 'softmax';
net.costFunction = 'cross';  
net.optim.none = false;     % overrides all optimizations
net.optim.lasso = false;
net.optim.ridge = true;
net.optim.momentum = false;
net.optim.dropout = false;
net.droprate = .85;
net.optim.early = false;
net.threshold = 12;

net.fit(false);
net.predict('test');
