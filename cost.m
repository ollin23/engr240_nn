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

% fprintf('size(target) : %d %d\n',size(target));
% fprintf('size(prediction) : %d %d\n',size(prediction));

switch self.costFunction
    case 'sparse_cross'
        if derivative
            J = prediction - target;
        else
            J = -sum(target .* log(prediction));
            J =  J  / size(target,1);
        end
    case 'cross'
        if derivative
            % tentative
            J = target - prediction;

        else
            %J = -sum(target .* log(prediction));
            J = target .* log(prediction) + (1-target) .* log(1-prediction);
            J = -J / length(target);
        end
    case 'hinge'
        if derivative
            % tentative, not actual derivative
            J = target - prediction;
        else
            J = max(0, 1-prediction .* target);
        end
    case {'kl', 'KL'}
        if derivative
            % tentative, not actual derivative
            J = (log(prediction ./ target) +1);
        else
            J = sum(prediction .* log(prediction ./ target));
        end
    case 'mse'
        if derivative
            J = (target - prediction) / length(target);
        else
            J = .5*sum((target - prediction).^2);
        end
end
 
% regularization
if ~(self.options.none)
    L1Norm = 0;
    L2Norm = 0;
    fn = fieldnames(self.layers);
    for k = 1:numel(fn)
        if self.options.ridge
            L2Norm= L2Norm + ...
                self.lambda * ...
                sum(sum(self.layers.(fn{k}).weights .^2));
        end
        if self.options.lasso
            L1Norm = ...
                self.lambda *...
                abs(sum(sum(self.layers.(fn{k}).weights)));
        end
    end
    
    J = J + L1Norm + L2Norm;

end

end