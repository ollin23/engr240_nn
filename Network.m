classdef Network
    % Network creates the neural network
    properties
        % parameters
        weights = {};
        memory = {};
        errors = {};
        bias = {};
        targetMatrix = [];
               
        % hyperparameters
        epochs;
        eta;
        batches;
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
            net.targetMatrix = eye(layers(end));
            
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
        function [c] = cost(prediction, actual, targetMatrix)
        end
        
        % update the weights
        function errors = update()
        end
    end
    
end