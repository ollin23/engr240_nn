classdef Network < handle
    % Network creates the neural network
    properties
        % internal parameters
        weights = {};       % matrices of weights between nodes
        memory = {};        % output from prior layer to current layer
        bias = {};          % bias
        transfer = '';      % transfer function type
        lastLayer = '';     % activation function for output layer
        costFunction = '';  % type of cost function
        
        % metric parameters
        errors = [];        % stores history of errors over the epochs
        accuracy;           % accuracy per batch
        images = [];        % structure for image data
        labels = [];        % structure for image labels
        encodedLabels = []; % structure for one hot encoded image labels
               
        % hyperparameters
        epochs;             % number of cycles through the training data
        eta;                % learning rate
        batches;            % number of batches for mini-batch training
    end
    methods
        % constructor 
        function net = Network(layers)
            if nargin == 0
                % can change to generalize; this specific design is for
                % treatment of the MNIST handwritten numbers dataset
                layers = [784, 38, 16, 10];
            end
            
            [w, b] = createNetwork(layers);

            % internal parameters
            net.weights = w;
            net.bias = b;
            net.transfer = 'tanh';
            net.lastLayer = 'softmax';
            net.costFunction = 'cross';
            
            % metric parameters
            net.errors = [0 0];
            net.accuracy = 0;
            net.images = [];
            net.labels = [];
            net.encodedLabels = [];
            
            % hyperparameters
            net.epochs = 1000;
            net.eta = .001;
            net.batches = round(2*sqrt(double(layers(1))));
        end

        function [h] = feedforward(layer, X)
        %feedforward returns the linear matrix from the given data
        % W: weight matrix
        % X: data from previous layer
        % b: bias vector
            h = feedforward(layer, net.weights, X, net.bias);
        end

        function [c] = cost(func, prediction, targets)
        %cost returns the cost/loss depending on the activation function
        %
        % func : the type of cost/loss function; string input
        % options -
        %   'cross'
        %       :: calculates the cross-entropy of between the prediction data and
        %          the labeled data
        %   'hinge'
        %       :: hinge loss used for classifiers; typically used for support
        %          vector machines (SVMs), but applicable elsewhere
        %   'KL'
        %       :: KullBack-Leibler Divergence, also called relative entropy
        %          is a measure of how the predicted probability distribution differs
        %          from the labeled data
        %   'MSE' (mean squared error)
        %       :: calculates the mean squared error the prediction data and
        %          the labeled data
            c = cost(func, prediction, targets);
        end
        
        function [a] = activate(h, func, derivative)
        %ACTIVATE takes the hypothesis vector, h, and transforms it into output for
        %the next layer of the neural net
        %
        % PARAMTER: h
        % The input vector.
        %
        % PARAMETER: func
        % A string indicating one of he four activation functions.
        % VALUES:
        %  - relu: Rectified Linear Unit (also known as ReLU). Returns the max of
        %       the input linear function or zero.
        %  - tanh: hyperbolic tangent. Transforms h to the range of (-1,1)
        %  - softmax: softmax function (see softmax notes). Used primarily for the
        %       output layer
        %  - sigmoid: the default activation function
        %
        % PARAMETER: derivative
        % The parameter derivative is a boolean which determines if the activation
        % function will use its derivative. The derivative of the activation
        % function is used to find the gradient for the node during
        % backpropagation.
        % VALUES:
        %  - true: used for backpropagation
        %  - false: used for feedforward
            a = activate(h, func, derivative);
        end
        
    end
end