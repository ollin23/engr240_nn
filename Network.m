classdef Network
    % Network creates the neural network
    properties
        % parameters
        weights = {};
        memory = {};
        errors = [];
        bias = {};
        transfer = '';
        lastLayer = '';
        costFunction = '';
               
        % hyperparameters
        epochs;     % number of cycles through the training data
        eta;        % learning rate
        batches;    % number of batches for mini-batch training
    end
    methods
        % constructor 
        function net = Network(layers)
            if nargin == 0
                layers = [784, 38, 16, 10];
            end
            
            [w, b] = createNetwork(layers);

            % parameters
            net.weights = w;
            net.bias = b;
            net.transfer = 'tanh';
            net.lastLayer = 'softmax';
            net.costFunction = 'cross';
            
            % hyperparameters
            net.epochs = 1000;
            net.eta = .001;
            net.batches = round(2*sqrt(double(layers(1))));
        end
        
        % network feedforward
        function [h] = feedforward(layer, X)
            h = feedforward(layer, net.weights, X, net.bias);
        end
        
        % activation functions
        % ReLU, tanh, softmax, and (default) sigmoid
        function [a] = activate(h, func, derivative)
            a = activate(h, func, derivative);
        end
        
        function [c] = cost(func, prediction, targets)
        % cost function
            c = cost(func, prediction, targets);
        end
        
        function trainNetwork(cycles, imgs, encLbls)
        % trainNetwork trains the neural net
        % PARAMETERS:
        % cycles - number of epochs in the training cycle
        % images - image data
        % encodedLabels - target label data
            
            for i = 1:cycles
                fprintf('Epoch %d :',i);
                J = train(imgs, encLbls, nn);
                J = mean(J);
                fprintf('\tError: %0.5f\n', J);
            end
        end
    end
end