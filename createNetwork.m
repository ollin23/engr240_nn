function [weights, bias] = createNetwork(layers, GPU)
%createNetwork produces a fully connected a neural net
%
%PARAMETERS
% layers - number of nodes in each separate layer
% GPU - enables/disables GPU acceleration
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
    H1 = layers(i);
    if i == 1
        H2 = H1;
    else
        H2 = layers(i-1);
    end
    b = sqrt(6) / sqrt(H1 + H2);
    if GPU
        weights{i} = -b + (2*b)*rand([layers(i+1) layers(i)],'gpuArray');
        bias(i) = gpuArray(1);
    else
        weights{i} = -b + (2*b)*rand([layers(i+1) layers(i)]);
        bias(i) = 1;
    end
end


end