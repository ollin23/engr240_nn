function [prediction] = feedforward2(self, X)
%feedforward returns the nonlinearized matrix from the given data
%
%PARAMETERS
% self.weights: weight matrix
% X: input data
% self.bias: bias vector
%
%OUTPUT
% h - input for next layer; transformed data

layers = length(self.weights);

for layer = 1:layers
    h =  X * self.weights{layer}'  + self.bias(layer);

    % save pre-transferred data in memory for backprop
    self.memory{layer} = h;
    
    % activation/transfer of weighted data
    if layer == layers
        transferFunction = self.lastLayer;
    else
        transferFunction = self.transfer;
    end
    
    
    a = activate(h, transferFunction, false);
            
    % store output as next layers input
    X = a;

    % save last layer output as prediction
    if layer == layers
        prediction = a;
    end
end

end