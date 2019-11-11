function train(self)
%train executes feedforward, backpropagation, and updates the biases and
%weights of neural network

    % size of the sample set
    sampleSize = length(self.images);


    % determine batch size
    % If batchSize = 1, then 
    batchSize = ceil(sampleSize / self.batches);
    layers = length(self.weights);
    
    % allocate space for error and accuracy
    J = 0;

    % Start the minibatches. If there is only one batch, there will only be
    % one loop
    for batch = 1:self.batches
        % clear out the memory of previous batches
        self.memory = {};
        batchSample = [];
        
        %fprintf('\nbatchSize : %d\n',batchSize);
        % designate start and end points for batch
        tmpStart = (batch * batchSize) - batchSize + 1;
        if tmpStart >= sampleSize
            start = stop +1;
        else
            start = tmpStart;
        tmpStop = batch * batchSize;
        if tmpStop >= sampleSize
            stop = sampleSize;
        else
            stop = tmpStop;
        end
        range = start:stop;
        
        distance = abs(stop - start + 1);
%         fprintf('distance : %d\n',distance);
%         fprintf('start stop : %d %d\n',start, stop);
        if distance < batchSize
%             fprintf('sumbitch\n');
            head = self.images(1:distance,:);
            tail = self.images(range,:);
            batchSample = [head; tail];
%             fprintf('size([tail]) :  %d %d\n',size([tail]));
%             fprintf('size(batchSample) :  %d %d\n',size(batchSample));
            batchEncodedLabels = [self.encodedLabels(1:distance,:);...
                self.encodedLabels(range,:)];
        else
            batchSample = self.images(range,:);
            batchEncodedLabels = self.encodedLabels(range,:);
        end
        % pull samples, encoded labels, and numeric labels
        % if the current batch size exceeds the last sample, pull the
        % remainder for the batch from the head of the sample set
        
        
        % Cycle through each sample in the batch
        for b = 1:batchSize
            input = batchSample(b,:);

            % Feedforward
            % Process each sample through every neural layer
            for i = 1:layers 
                h = feedforward(i, self.weights, input, self.bias);

                % Use different transfer function for last layer
                if i == layers
                    func = self.lastLayer;
                else
                    func = self.transfer;
                end
                a = activate(h,func,false);

                % save data in memory
                self.memory{b,i} = h;

                % store output data as input in the next layer
                input = a;
            end
        end

        % calculate error at end of batch
        [rows, cols] = size(self.memory);
        prediction = reshape([self.memory{:,cols}],[],rows)';

        J = sum(cost(self.costFunction, prediction, batchEncodedLabels));
        
        self.accuracy = sum(dot(prediction,batchEncodedLabels));
        self.errors = [self.errors; J, self.accuracy];

%         fprintf('\nRANGE:  %d %d\n',start, stop);
        % backpropagation
        [newWeights, newBias] = ...
            backprop(batchSample, batchEncodedLabels, prediction, b, self);
        update(self, newWeights, newBias);
    end
end
