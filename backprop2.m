function [weightDelta, biasDelta] = backprop2(self, prediction, target, input)

layers = length(self.weights);

weightDelta = cell(1,layers);
biasDelta = zeros(layers);

% gradient for hidden layers
for layer = layers:-1:1

    h = self.memory{layer};
        
    % last to output layer is different
    if layer == layers
        gradient = prediction - target;
        h = self.memory{layer-1};
        a = activate(h,self.transfer,true);
    else
        gradient = gradient * self.weights{layer+1};
        gradient = activate(h,self.transfer,true) .* gradient;
        if layer-1 == 0
            a = input;
        else
            a = activate(self.memory{layer-1},self.transfer,false);
        end
    end
    
    delta = gradient' * a;
    deltaBias = mean(mean(gradient') * self.bias(layer));
    
    switch self.optimization
        case 'none'
            delta = self.eta * delta;
            deltaBias = self.eta * deltaBias;
    end
    
    weightDelta{layer} = -delta;
    biasDelta(layer) = -deltaBias;
end
    
end % end of function
    
