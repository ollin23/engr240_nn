function [weights, bias] = createNetwork(layers)
%createNetwork produces a fully connected a neural net
%
%USAGE
%  layers = [784, 300, 150, 50, 10];
%  [weights, biases] = createNetwork(layers);
%
%PARAMETERS
% layers - number of nodes in each separate layer
%
% OUTPUT
% weights : the weights connecting the layers and nodes
% bias : layer bias
%

% seed the RNG with Mersenne Twister and a seed of 1 for reproducibility
rng(42,'twister');
weights = {};
bias = [];

% initialization schema taken from Golorot and Bengio, 2010
for i = 1:(length(layers)-1)
    H1 = double(layers(i));
    if i == 1
        H2 = double(H1);
    else
        H2 = double(layers(i-1));
    end
    b = sqrt(6.0) / sqrt(H1 + H2);
    weights{i} = -b + (2*b)*rand([layers(i+1) layers(i)]);
    bias(i) = 1;
end


end