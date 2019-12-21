function update2(self, deltaWeights, deltaBias)
% update updates the weights and biases with the changes calculated during
% backpropagation
%
% PARAMETERS
% self
% deltaWeights
% deltaBias


for l = 1:length(self.weights)
    self.weights{l} = self.weights{l} - deltaWeights{l};
    self.bias(l) = self.bias(l) - deltaBias(l);
end

end
