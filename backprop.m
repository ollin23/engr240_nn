function [] = backprop(layers, batchEncodedLabels, prediction, batch)
%backprop executes the backpropagation algorithm on the network
%
%
    b = batch;
    
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