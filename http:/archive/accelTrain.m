function [err, acc, prec, recall, R2] = accelTrain(self)
%gpuTrain executes feedforward, backpropagation, and updates the biases and
% weights of neural network

    % size of the sample set
    [sampleSize,~] = size(self.images.training);
    ybar = mean(self.images.tngEncLabels);

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
    runningRecall = [];
    runningR2 = [];

    epochError = [];
    epochAccuracy = [];
    epochPrecision = [];
    epochRecall = [];
    epochR2 = [];
    
    % helper code for displaying percentage complete
    k = 2;
    if sampleSize > 1000
        k = sampleSize / 1000;
    end
    counter = 0;
    
    inputs = self.images.training;
    targets = self.images.tngEncLabels;
    
    % * * * * * * * * * * * * * * * * * * * 
    % begin cycling through the samples
    % * * * * * * * * * * * * * * * * * * * 
    for sample = 1:sampleSize
        counter = counter + 1;
        self.memory = {};
        target = targets(sample,:);
        input = inputs(sample,:);
        %label = self.labels(sample);
        
        % * * * * * * * * * * *
        %      FEEDFORWARD
        % * * * * * * * * * * *
        prediction = feedforward2(self, input);
        
        % * * * * * * * * * * * *
        % Calculate Error
        % * * * * * * * * * * * *
        J = cost(self, prediction, target, false);

        % calculate accuracy and precision
        [acc, prec, recall, R2] = metrics(prediction, target, ybar);

        % append precision, accuracy, and error
        runningPrecision = cat(1,prec,runningPrecision);
        runningAccuracy = cat(1,acc,runningAccuracy);
        runningError = cat(1,J,runningError);
        runningRecall = cat(1,recall,runningRecall);
        runningR2 = cat(1,R2,runningR2);

        %* * * * * * * * * * * * * * * * * * * * *
        %             BACK PROPAGATION
        %* * * * * * * * * * * * * * * * * * * * *
        [w, b] = backprop2(self, prediction, target, input);
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
            recall = mean(runningRecall);
            R2 = mean(runningR2);

            epochPrecision = cat(1,prec,epochPrecision);
            epochAccuracy = cat(1,acc,epochAccuracy);
            epochError = cat(1,err,epochError);
            epochRecall = cat(1,recall,epochRecall);
            epochR2 = cat(1,R2,epochR2);
            
            update2(self, newWeights, newBias);
            
            % reset batch memory
            newWeights = cell(batchSize, layers);
            newBias = zeros(batchSize, layers);
            counter = 0;
            runningPrecision = [];
            runningAccuracy = [];
            runningError = [];
            runningRecall = [];
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
    %self.memory = memory;
    
    
    % return error, accuracy, and precision
    err = mean(epochError);
    acc = mean(epochAccuracy);
    prec = mean(epochPrecision);
    recall = mean(epochRecall);
    R2 = mean(epochR2);
    
    self.errors = cat(1,err,self.errors);
    self.accuracy = cat(1,acc,self.accuracy);
    self.precision = cat(1,prec,self.precision);
    self.R2 = cat(1,R2,self.R2);

end % end of function
