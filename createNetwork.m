function [weights, bias] = createNetwork(layers)
%createNetwork produces a fully connected a neural net
%
% weights : the network 
%
%

% seed the RNG with Mersenne Twister and a seed of 1 for reproducibility
rng(1,'twister');
weights = {};
bias = [];

% designate the weight to each node in each layer with a random real number
% from the uniform distribution between (0,1) then halved
for i = 1:(length(layers)-1)
    weights{1,i} = randn([layers(i+1) layers(i)]) *.5;
    bias(i) = 1;
end

end