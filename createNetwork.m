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

% % designate the weight to each node in each layer with a random real number
% % from the uniform distribution between (0,1) then halved
% for i = 1:(length(layers)-1)
%     weights{i} = rand([layers(i+1) layers(i)]) *.5;
%     bias(i) = 1;
% end

% initialization schema taken from Golorot and Bengio, 2010
for i = 1:(length(layers)-1)
    H1 = layers(i);
    if i == 1
        H2 = H1;
    else
        H2 = layers(i-1);
    end
    b = sqrt(6) / sqrt(H1 + H2);
    weights{i} = -b + (2*b)*rand([layers(i+1) layers(i)]);
    bias(i) = 1;
end


end