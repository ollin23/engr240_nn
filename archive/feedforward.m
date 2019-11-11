function [h] = feedforward(layer, W, X, b)
%feedforward returns the nonlinearized matrix from the given data
% W: weight matrix
% X: data from previous layer
% b: bias vector

h = X * W{layer}' + b(layer);

end
