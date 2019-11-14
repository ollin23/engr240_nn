function train2(self)
%train executes feedforward, backpropagation, and updates the biases and
% weights of neural network

    % size of the sample set
    [sampleSize,~] = size(self.images);

    % determine batch size
    % If batchSize == 1 or 0, then batchSize = 1
    if self.batches == 1 || self.batches == 0
        batchSize = sampleSize;
    elseif self.batches > sampleSize
        batchSize = sampleSize;
    else
        batchSize = ceil(sampleSize / self.batches);
    end
    layers = length(self.weights);
    
    % allocate space for error, new weights, and bias
    J = 0;
    newWeights = cell(batchSize, layers);
    newBias = zeros(batchSize,layers);
    
    k = 2;
    if sampleSize > 1000
        k = sampleSize / 1000;
    end
    counter = 0;
    % begin cycling through the samples
    for sample = 1:sampleSize
        counter = counter + 1;
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

        J = cost('cross', prediction, target);
        acc = dot(target,prediction);
                    
        self.accuracy = cat(1,acc,self.accuracy);
        self.errors = cat(1,J,self.errors);

        [w, b] = backprop2(self, prediction, target, self.images(sample,:));
        b = mean(b,2);
        newWeights(counter,:) = w;
        newBias(counter,:) = b;

        % update according to batchSize
        % the below enables the use of SGD, batch, and various sizes of
        % minibatches
        if mod(sample,batchSize) == 0 && batchSize > 1 && sample ~= 1
            newWeights = mean(cellfun(@(x)mean(x(:,1)),newWeights));
            newBias = mean(newBias);
            
            update2(self, newWeights, newBias);
            
            % reset batch memory
            %newWeights = {};
            newWeights = cell(batchSize, layers);
            newBias = zeros(batchSize, layers);
            counter = 0;
        end
        
        % keep track of percentage done
        if mod(sample,k) == 0
            percent = 100 * (sample/sampleSize);
            strPer = ['  ' num2str(percent) '%%'];
            if percent < 100
                strCR = [repmat('\b',1,length(strPer)-1)];
            else
                strCR = [repmat('\b',1,length(strPer)-1) '\b\b'];
                strPer = [strPer '\n'];
            end
            fprintf([strCR strPer]);
        end
    end  % end of sample set
end % end of function