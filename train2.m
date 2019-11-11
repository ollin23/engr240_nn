function train2(self)
%train executes feedforward, backpropagation, and updates the biases and
% weights of neural network

    % size of the sample set
    sampleSize = length(self.images);

    % determine batch size
    % If batchSize == 1 or 0, then batchSize = 1
    if self.batches == 1 || self.batches == 0
        batchSize = sampleSize;        
    else
        batchSize = ceil(sampleSize / self.batches);
    end
    layers = length(self.weights);
    
    % allocate space for error, new weights, and bias
    J = 0;
    newWeights = cell(batchSize, layers);
    newBias = zeros(batchSize,layers);
    
    % begin cycling through the samples
    for sample = 1:sampleSize
        self.memory = {};
        input = self.images(sample,:);
        target = self.encodedLabels(sample,:);
        %feedforward
        for l = 1:layers
            h = feedforward2(self, l, input);
            
            % activation/transfer of weighted data
            if l == layers
                transferFunction = self.lastLayer;
            else
                transferFunction = self.transfer;
            end
            a = activate(h, transferFunction, false);
            
            % save pre-transferred data in memory for backprop
            self.memory{l} = h;
            
            % store output as next layers input
            input = a;
            
        end % end of layers

        % calculate error
        [rows, cols] = size(self.memory);
        prediction = reshape([self.memory{:,cols}],[],rows)';

        J = J + sum(cost(self.costFunction, prediction, target)) / sample;
        
        self.accuracy = sum(dot(prediction,target)) / length(self.accuracy);
        self.errors = [self.errors, J];
    
        %fprintf('\nSample : %d\n',sample);
        % backprop
        [w, b] = backprop2(self, self.images(sample,:));

        newWeights(sample,:) = w;
        newBias(sample,:) = b(1,:);
        
        % update according to batchSize
        % the below enables the use of SGD, batch, and various sizes of
        % minibatches
        if mod(batchSize,sample) == 0
            update2(self, newWeights, newBias);
            newWeights = {};
        else
            [~, cols] = size(newWeights);
            for c = 1:cols
                newWeights{c} = mean(cat(1,newWeights{:,1}));
                newBias(c) = mean(newBias(c));
            end
        end
        
        % keep track of percentage done
        percent = 100 * (sample/sampleSize);
        strPer = [num2str(percent) '%%'];
        if percent < 100
            strCR = [repmat('\b',1,length(strPer)-1)];
        else
            strCR = [repmat('\b',1,length(strPer)-1) '\b\b'];
            strPer = [strPer '\n'];
        end
        fprintf([strCR strPer]);
        
        
        
    end  % end of sample set
end % end of function