function [h] = feedforward2(self, layer, X)
%feedforward returns the nonlinearized matrix from the given data
%
%PARAMETERS
% self.weights: weight matrix
% X: input data
% self.bias: bias vector
%
%OUTPUT
% h - input for next layer; transformed data

h =  X * self.weights{layer}'  + self.bias(layer);

end