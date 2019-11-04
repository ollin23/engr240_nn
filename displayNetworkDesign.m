function [] = displayNetworkDesign(layers)
%displayNetworkDesign displays the design of the network

[~, layerCount] = size(layers);

fprintf('The network shape is:\n');
for i = 1:layerCount
    if i == 1
        fprintf('\tlayer %d: input layer, %d nodes\n', i, layers(i));
    elseif i == layerCount
        fprintf('\tlayer %d: output layer, %d nodes\n\n',...
            i, layers(layerCount));
    else
        fprintf('\tlayer %d: hidden layer %d, %d nodes\n',...
            i, i-1, layers(i));
    end
end
end
