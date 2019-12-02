function [J] = cost(self, prediction, target, derivative)
%cost returns the cost/loss depending on the activation function
%
% cost(func, prediction, target)
% PARAMETERS
% prediction :
%   the output vector from the final layer
%
% target :
%   the label against which the prediction is tested
%
% func : the type of cost/loss function; string input
%   'cross'
%       :: calculates the cross-entropy of between the prediction data and
%          the labeled data
%   'hinge'
%       :: hinge loss used for classifiers; typically used for support
%          vector machines (SVMs), but applicable elsewhere
%   'KL'
%       :: KullBack-Leibler Divergence, also called relative entropy
%          is a measure of how the prediction probability distribution
%          differs from the labeled data
%   'MSE' 
%       :: Mean Squared Error, calculates the mean squared error the
%          prediction data and the labeled data
%
% derivative :
%   boolean value; determines if the cost derivative should be used
%

switch self.costFunction
    case 'cross'
        if derivative
            J = (target./prediction) + (1-target)/(1-prediction);
            J = -J;
        else
            J = -sum(target .* log(prediction));
        end
    case 'hinge'
        if derivative
            % tentative, not actual derivative
            J = target - prediction;
        else
            J = max(0, 1-prediction .* target);
        end
    case 'kl'
        if derivative
            % tentative, not actual derivative
            J = target - prediction;
        else
            J = sum(prediction .* log(prediction ./ target));
        end
    case 'mse'
        if derivative
            J = target - prediction;
        else
            J = .5*sum((target - prediction).^2);
        end
end


% regularization
if (self.optim.none == false)
    if self.optim.lasso
        L1norm = 0;
        for w = 1:length(self.weights)
            L1norm = L1norm + norm(self.weights{w});
        end
        J = J + ...
            (self.lambda / numel(self.weights)) * ...
            L1norm;
            % sum(cellfun(@(x)sum(x(:)),self.weights));
    end
    if self.optim.ridge
        L2norm = 0;
        for w = 1:length(self.weights)
            L2norm = L2norm + norm(self.weights{w}.^2);
        end
        J = J + ...
            (self.lambda / length(self.labels)) * ...
            L2norm;
            % sum(cellfun(@(x)sum(x(:).^2),self.weights));
    end
end


end