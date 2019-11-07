function [] = backprop(batchEncodedLabels, prediction, batch, nn)
%backprop executes the backpropagation algorithm on the network
%PARAMETERS
% layers - number of layers for the algorithm to propagate through
% batchEncodedLabels - the target labels for the current batch
% prediction - the predicted result
% batch - reference for the current batch
% nn - the neural network
%
    b = batch;
    layers = length(nn.weights);
    % move backwards through the layers
    for i = layers:-1:1
        if i == layers
            func = nn.lastLayer;
            dJ = sum(batchEncodedLabels - prediction);
        else
            func = nn.transfer;
            dJ = nn.weights{i+1};
        end
        h = nn.memory{b,i};
        da = activate(h,func,true);
        deltaWeights = (dJ .* da) * nn.memory{b,i}';
        deltaBias = (dJ .* da) * nn.bias(i);
        
        % update weights and bias
        nn.weights{i} = nn.weights{i} - nn.eta * sum(deltaWeights)';
        nn.bias(i) = nn.bias(i) - nn.eta * sum(mean(deltaBias));
    end
end