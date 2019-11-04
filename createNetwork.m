function [weights, bias, errors] = createNetwork(layers)
%createNetwork produces a fully connected a neural net
% weights : the network 
%
%
rng(1,'twister');
weights = {};
bias = [];

for i = 1:(length(layers)-1)
    weights{1,i} = randn([layers(i+1) layers(i)]) *.5;
    bias(i) = 1;
    errors{1,i} = ones(1,layers(i));
end


end