function [weightDelta, biasDelta] = backprop2(self, input)

layers = length(self.weights);

%gradient for output layer
h = self.memory{layers};
gradient = (activate(h,self.lastLayer,true));

weightDelta = cell(1,layers);
biasDelta = zeros(layers);


% gradient for hidden layers
for i = layers:-1:1

    W = self.weights{i};
    
    % retrieve previous layer's non-activated output
    if (i-1) == 0
        h = input;
        h1 = 0;
    else
        h = self.memory{i};
        h1 = self.memory{i-1};
    end

    % calculate derivatives
    da = (activate(h,self.transfer,true));

    % calculate gradient
    if i == 1
        deltaBias = mean(mean(gradient') * self.bias(i));
        gradient = gradient' * da;
    else
        if size(gradient) == size(da)

        elseif size(gradient) < size(da)
            [x] = abs(size(gradient) - size(da));
            gradient = cat(1,repmat(gradient,x(1),1),gradient);
        else
            [x] = abs(size(da) - size(gradient));
            da = cat(1,repmat(da,x(1),1),da);
        end
        gradient = gradient .* da;
        deltaBias = mean(mean(gradient) * self.bias(i));
    end
    
    % apply optimization feature
    switch self.optimization
        case 'none'
            delta = self.eta * gradient'*h1;
    end

    % gradient for next layer
    if i ~= 1
        gradient = (gradient * W);
        
    % make sure the delta is facing the right direction for the last layer
    else
        delta = delta';
    end
   
    % output deltas for update
    weightDelta{i} = delta;
    biasDelta(i) = deltaBias;
    
end % end of layers
    
    % calculate multiple summations
    function [y] = allsum(x)
        y = sum(x);
        if length(y) == 1
            y = y;
        else
            y = allsum(y);
        end
    end
   
end