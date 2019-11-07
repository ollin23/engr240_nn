function [J] = train(images, encodedLabels, nn)
%train executes feedforward, backpropagation, and updates the biases and
%weights of neural network

    samples = length(images);
    batchSize = ceil(samples / nn.batches);
    layers = length(nn.weights);
    J = [];

    for batch = 1:nn.batches
        nn.memory = {};

        % designate start and end points for batch
        start = (batch * batchSize - batchSize) + 1;
        stop = (batch * batchSize);
        range = start:stop;

        % pull samples, encoded labels, and numeric labels
        batchSample = images(range,:);
        batchEncodedLabels = encodedLabels(range,:);

        % for each sample in batch
        for b = 1:batchSize
            input = batchSample(b,:);

            % process each sample through every neural layer
            for i = 1:layers 
                h = feedforward(i, nn.weights, input, nn.bias);

                % use different transfer function for last layer
                if i == layers
                    func = nn.lastLayer;
                else
                    func = nn.transfer;
                end
                a = activate(h,func,false);

                % save data in memory
                nn.memory{b,i} = a;

                % use processed data for input in the next layer
                input = a;
            end
        end

        % calculate the error at end of batch
        [rows, cols] = size(nn.memory);
        prediction = reshape([nn.memory{:,cols}],...
            length(batchEncodedLabels),rows)';

        J = [J; mean(cost(nn.costFunction, prediction, batchEncodedLabels))];
        nn.errors = [nn.errors; J];

        % backpropagation
        backprop(batchEncodedLabels, prediction, b, nn);
    end
end
