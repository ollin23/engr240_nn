function update(self, newWeights, newBias)
% update updates the weights and biases with the changes calculated during
% backpropagation

for i = 1:length(self.weights)
    self.weights{i} = newWeights{i};
    self.bias(i) = newBias(i);
end

end