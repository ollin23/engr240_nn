classdef Network
    % Network creates the neural network
    properties
        % parameters
        weights = {};
        memory = {};
        errors = [];
        bias = {};
               
        % hyperparameters
        epochs;  % number of cycles through the training data
        eta; % learning rate
        batches; % number of batches for mini-batch training
    end
    methods
        % constructor 
        function net = Network(layers)
            if nargin == 0
                layers = [784, 38, 16, 10];
            end
            
            [w, b, e] = createNetwork(layers);

            % parameters
            net.weights = w;
            net.bias = b;
            net.errors = e;
            
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
        
        % network backpropagation
        function []= backprop()
        end
        
        % cost function
        function [c] = cost(func, prediction, targets)
            c = cost(func, prediction, targets);
        end
        
        % find the deltas for the weights
        function deltas = getDeltas()
        end
        
        
    end
    
end