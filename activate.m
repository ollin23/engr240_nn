function a = activate(h, func, derivative)
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
    switch func
        case 'relu'
            if derivative
                if h < 0
                    a = 0;
                else
                    a = 1;
                end
            else
                a = max(0,h);
            end
        case 'tanh'
            if derivative
                a = sech(h).^2;
            else
                a = tanh(h);
            end
        case 'softmax'
            if derivative
                a = softmax(h) .* (1 - softmax(h));
            else
                a = softmax(h);
            end
        otherwise
            % default is the Sigmoid function
            if derivative
                a = sigmoid(h) .* (1 - sigmoid(h));
            else
                a = sigmoid(h);
            end
    end
end