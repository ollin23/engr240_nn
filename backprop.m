function backprop(input, batchEncodedLabels, prediction, batch, self)
%backprop executes the backpropagation algorithm on the network
%
%PARAMETERS
% layers - number of layers for the algorithm to propagate through
% batchEncodedLabels - the target labels for the current batch
% prediction - the predicted result
% batch - reference for the current batch
% nn - the neural network
%
    b = batch;
    layers = length(self.weights);
    
    % move backwards through the layers
    for i = layers:-1:1
        
        % Backprop for final output layer
        if i == layers
            %func = nn.lastLayer;
            delta = sum(batchEncodedLabels - prediction);
            prior = self.memory{b,i};
            gradient  = prior * delta';
            
        % Backprop for non-final layers
        else
            func = self.transfer;
            
            if (i-1) == 0
                h = input;
            else
                h = [self.memory{:,i}];
                h = sum(reshape(h,b,length(self.memory{b,i})));
            end
            oldWeights = self.weights{i};
            delta = mean(activate(h,func,true));            
            gradient = delta * oldWeights';
        end
        %deltaWeights = (dJ .* da) * nn.memory{b,i}';
        nablaBias = gradient .* self.bias(i);
        
        % update weights and bias
        self.weights{i} = self.weights{i} + self.eta * gradient';
        self.bias(i) = self.bias(i) - self.eta * sum(mean(nablaBias));

    end
end