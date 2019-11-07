function [] = train(batches, images, labels, encodedLabels) 
samples = length(images);
%batches = 20;
batchSize = ceil(samples / batches);
layers = length(nn.weights);
J = [];

% I = images(1:samples,:);
% eL = encodedLabels(1:samples,:);
% La = labels(1:samples);
% 
% sampleCounter = 1;

for batch = 1:batches
    fprintf('BATCH %d',batch)
    memory = {};
    
    % designate start and end points for batch
    start = (batch * batchSize - batchSize) + 1;
    stop = (batch * batchSize);
    range = start:stop;
    
    % pull samples, encoded labels, and numeric labels
    batchSample = images(range,:);
    batchEncodedLabels = encodedLabels(range,:);
    batchLabels = labels(range);
    
    % for each sample in batch
    for b = 1:batchSize
        input = batchSample(b,:);
        
        % process each sample through every neural layer
        for i = 1:layers 
            h = feedforward(i, nn.weights,input,nn.bias);
            
            % use softmax at the end otherwise, use tanh
            if i == layers
                func = 'softmax';
            else
                func = 'tanh';
            end
            a = activate(h,func,false);
            
            % save data in memory
            nn.memory{b,i} = a;
            
            % use processed data for input in the next layer
            input = a;
        end
    end
    
    % calculate the error at end of minibatch
    [rows, cols] = size(memory);
    prediction = reshape([memory{:,cols}],length(batchEncodedLabels),rows)';
    J = [J; cost('cross', prediction, batchEncodedLabels)];
    
    
    % calculate deltas
    for i = layers:-1:1
        if i == layers
            func = 'softmax';
            dJ = sum(batchEncodedLabels - prediction);
        else
            func = 'tanh';
            dJ = nn.weights{i+1};
        end
        h = memory{b,i};
        da = activate(h,func,true);
        deltaWeights = (dJ .* da) * memory{b,i}';
        deltaBias = (dJ .* da) * nn.bias(i);
        nn.weights{i} = nn.weights{i} - nn.eta * sum(deltaWeights)';
        nn.bias(i) = nn.bias(i) - nn.eta * sum(mean(deltaBias));
    end

end
