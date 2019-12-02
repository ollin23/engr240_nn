function [weightDelta, biasDelta] = backprop2(self, prediction, target, input)

layers = length(self.weights);

weightDelta = cell(1,layers);
biasDelta = zeros(layers);

% gradient for hidden layers
for layer = layers:-1:1

    h = self.memory{layer};
        
    % last to output layer is different
    if layer == layers
        gradient = cost(self, prediction, target, true);
        h = self.memory{layer-1};
        a = activate(h,self.transfer,true);
    else
        gradient = gradient * self.weights{layer+1};
        a = activate(h,self.transfer,true);
        gradient = a .* gradient;

        if layer-1 == 0
            a = input;
        else
            a = activate(self.memory{layer-1},self.transfer,false);
        end
    end
%     fprintf('layers %d\n',layer);
%     fprintf('size(gradient) = %d %d\n',size(gradient));
%     fprintf('size(a) = %d %d\n',size(a));
    
    delta = gradient' * a;
    deltaBias = mean(mean(gradient') * self.bias(layer));
    
    % * * * * * * * * * * * * * * * * 
    %        OPTIMIZATIONS
    % * * * * * * * * * * * * * * * * 
    delta = self.eta * delta;
    deltaBias = self.eta * deltaBias;
    if ~(self.optim.none)
        if self.optim.lasso
            delta = delta + ...
                self.lambda * norm(self.weights{layer}) / ...
                numel(self.weights{layer});
        end
        if self.optim.ridge
            delta = delta + ...
                self.lambda * norm(self.weights{layer}.^2) / ...
                numel(self.weights{layer});
        end
        if self.optim.momentum && exist('self.oldDeltas', 'var')
            delta = delta + self.mu * self.oldDeltas{layer};
        end
    end
    
    weightDelta{layer} = delta;
    deltaBias = gather(deltaBias);
    biasDelta(layer) = deltaBias;
    
    % store delta for use with momentum in next sample
    self.oldDeltas{layer} = delta;
end
    
end % end of function
    
