classdef Network < handle
    % Network creates the neural network
    properties
        % internal parameters
        weights = {};       % matrices of weights between nodes
        memory = {};        % output from prior layer to current layer
        bias = {};          % bias
        oldDeltas = {};     % deltas used in last 
        
        % metric parameters
        errors = [];        % stores history of errors over the epochs
        accuracy = [];      % running accuracy
        precision = [];     % running precision
        
        % image parameters
        images = [];        % structure for image data
        labels = [];        % structure for image labels
        encodedLabels = []; % structure for one hot encoded image labels
               
        % hyperparameters
        epochs;             % number of cycles through the training data
        eta;                % learning rate
        lambda;             % regularization hyperparameter
        mu;                 % momentum hyperparameter
        droprate;           % drop rate for use with dropout
        dropmasks = {};     % the masks for the dropout layers
        batches;            % number of batches for mini-batch training
        transfer = '';      % transfer function type
        lastLayer = '';     % activation function for output layer
        costFunction = '';  % type of cost function
        
        % NOTES:    (1) none overrides all other optimization techniques
        %           (2)ADAgrad, and RMSprop are mutually exclusive
        optim = ...
            struct('none', true,...     % no optimizations used
                   'ridge',false,...    % L1 regularization
                   'lasso',false,...    % L2 regularization
                   'momentum',false,... % enables gradient momentum
                   'dropout',false,...  % enables random dropout
                   'ADAgrad', false,... % enables ADAgrad
                   'RMSprop', false,... % enables RMSprop
                   'early', false);     % enables early termination
    
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
            
            % allocate space for oldDeltas
            for i = 1:length(w)
                net.oldDeltas{i} = zeros(size(w{i}));
            end
            
            % metric parameters
            net.errors = [];
            net.accuracy = [];
            net.precision = [];
            
            % image parameters
            net.images = [];
            net.labels = [];
            net.encodedLabels = [];
            
            % hyperparameters
            net.epochs = 300;
            net.eta = .0001;
            net.lambda = .01;
            net.mu = .5;
            net.batches = 64;
            net.droprate = .8;
        end

        function [h] = feedforward(self, layer, X)
        %feedforward returns the linear matrix from the given data
        % W: weight matrix
        % X: data from previous layer
        % b: bias vector
            h = feedforward2(self, layer, X);
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