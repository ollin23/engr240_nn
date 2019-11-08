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

        % designate start and end points for batch
        start = (batch * batchSize) - batchSize + 1;
        stop = batch * batchSize;
        range = start:stop;

        % pull samples, encoded labels, and numeric labels
        batchSample = self.images(range,:);
        batchEncodedLabels = self.encodedLabels(range,:);

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
                self.memory{b,i} = a;

                % store output data as input in the next layer
                input = a;
            end
        end

        % calculate error at end of batch
        [rows, cols] = size(self.memory);
        prediction = reshape([self.memory{:,cols}],...
            length(batchEncodedLabels),rows)';

        J = sum(cost(self.costFunction, prediction, batchEncodedLabels));
        self.accuracy = sum(dot(prediction,batchEncodedLabels));
        self.errors = [self.errors; J, self.accuracy];

        % backpropagation
        backprop(input, batchEncodedLabels, prediction, b, self);
    end
end
