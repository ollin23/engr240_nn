function [err, acc, prec, R2] = train2(self)
%train executes feedforward, backpropagation, and updates the biases and
% weights of neural network

    % size of the sample set
    [sampleSize,~] = size(self.images);
    ybar = mean(self.encodedLabels);

    % determine batch size
    % self.batches == 0, then batchSize == 1 , Stochastic Gradient Descent
    % self.batches == 1, batchSize == 1000, Batch Gradient Descent
    if self.batches == 0
        batchSize = 1000;
    elseif self.batches > sampleSize
        batchSize = mod(self.batches,sampleSize);
    else
        batchSize = ceil(sampleSize / self.batches);
    end
    layers = length(self.weights);
    
    % allocate space for error, new weights, new biases, running accuracy
    % and running precision
    newWeights = cell(batchSize, layers);
    newBias = zeros(batchSize,layers);
    runningError = [];
    runningAccuracy = [];
    runningPrecision = [];
    runningR2 = [];

    epochError = [];
    epochAccuracy = [];
    epochPrecision = [];
    epochR2 = [];
    
    % helper code for displaying percentage complete
    k = 2;
    if sampleSize > 1000
        k = sampleSize / 1000;
    end
    counter = 0;
    
    % * * * * * * * * * * * * * * * * * * * 
    % begin cycling through the samples
    % * * * * * * * * * * * * * * * * * * * 
    for sample = 1:sampleSize
        counter = counter + 1;
        self.memory = {};
        input = self.images(sample,:);
        target = self.encodedLabels(sample,:);
        %label = self.labels(sample);

        % * * * * * * * * * * *
        %      FEEDFORWARD
        % * * * * * * * * * * *
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
            
            % save last layer output as prediction
            if l == layers
                prediction = a;
            end
        end % end of layers
        
        self.longmemory = cat(1,prediction,self.longmemory);
        
        % * * * * * * * * * * * *
        % Calculate Error
        % * * * * * * * * * * * *
        J = cost(self.costFunction, prediction, target, false);

        if ~(self.optim.none)
            if self.optim.lasso
                L1norm = 0;
                for w = 1:length(self.weights)
                    L1norm = L1norm + norm(self.weights{w});
                end
                J = J + ...
                    (self.lambda / numel(self.weights)) * ...
                    L1norm;
                    % sum(cellfun(@(x)sum(x(:)),self.weights));
            end
            if self.optim.ridge
                L2norm = 0;
                for w = 1:length(self.weights)
                    L2norm = L2norm + norm(self.weights{w}.^2);
                end
                J = J + ...
                    (self.lambda / length(self.labels)) * ...
                    L2norm;
                    % sum(cellfun(@(x)sum(x(:).^2),self.weights));
            end
        end

        % calculate accuracy and precision
        p = round(normalize(prediction));
        
        % [~,c] = max(prediction);
        % oPred = oneHotEncoding(1:length(target));
        % p = oPred(c,:);

        positives = nnz(p);
        if positives > 0
            diff = (target - p);
            FP = nnz(diff<0);
            FN = nnz(diff>0);
            TN = sum(p == diff);
            TP = abs(1-FN);
            %pLabel = mod(find(p>0),length(target));
        else
            FP = 0;
            TP = 0;
            FN = 1;
            TN = 9;
            %pLabel = -1;
        end

        acc = (TP + FP)/(FP + TP + FN + TN);
        prec = TP / (TP + FP);
        SSE = sum((target - prediction).^2);
        SST = sum((target - ybar.^2));
        R2 = 1 - SSE/SST;

        % append precision, accuracy, and error
        runningPrecision = cat(1,prec,runningPrecision);
        runningAccuracy = cat(1,acc,runningAccuracy);
        runningError = cat(1,J,runningError);
        runningR2 = cat(1,R2,runningR2);

        [w, b] = backprop2(self, prediction, target, self.images(sample,:));
        b = mean(b,2);
        newWeights(counter,:) = w;
        newBias(counter,:) = b;
        
        % * * * * * * * * * * * * * * * * * * * * *
        %      UPDATE PARAMETERS AND METRICS
        % * * * * * * * * * * * * * * * * * * * * *
        % update according to batchSize
        % the below enables the use of SGD (batchSize == 1),
        % Batch Gradient Descent (batchSize = sampleSize),
        % and various in between sizes, ak Mini-batch Gradient Descent
        if mod(sample,batchSize) == 0
            if batchSize > 1
                 for i = 1:layers
                    tmp = newWeights(:,i);
                    newWeights{i} = (mean(tmp{i}));
                    newBias(i) = mean(newBias(i));
                 end
            end
            
            % newWeights = mean(cellfun(@(x)sum(x(:,1).^2),newWeights));
            err = mean(runningError);
            acc = mean(runningAccuracy);
            prec = mean(runningPrecision);
            R2 = mean(runningR2);
                        
            epochPrecision = cat(1,prec,epochPrecision);
            epochAccuracy = cat(1,acc,epochAccuracy);
            epochError = cat(1,err,epochError);
            epochR2 = cat(1,R2,epochR2);
            
            update2(self, newWeights, newBias);
            
            % reset batch memory
            newWeights = cell(batchSize, layers);
            newBias = zeros(batchSize, layers);
            counter = 0;
            runningPrecision = [];
            runningAccuracy = [];
            runningError = [];
            runningR2 = [];
        end
        
        % keep track of percentage done
        if mod(sample,k) == 0
            percent = 100 * (sample/sampleSize);
            strPer = ['  ' num2str(percent) '%%'];
            if percent < 100
                strCR = repmat('\b',1,length(strPer)-1);
            else
                strCR = [repmat('\b',1,length(strPer)-1) '\b\b'];
                strPer = [strPer '\n'];
            end
            fprintf([strCR strPer]);
        end
    end  % end of sample set
    
    % return error, accuracy, and precision
    err = mean(epochError);
    acc = mean(epochAccuracy);
    prec = mean(epochPrecision);
    R2 = mean(epochR2);
    
    self.errors = cat(1,err,self.errors);
    self.accuracy = cat(1,acc,self.accuracy);
    self.precision = cat(1,prec,self.precision);
    self.R2 = cat(1,R2,self.R2);

end % end of function