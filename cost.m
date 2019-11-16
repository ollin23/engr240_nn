function [J] = cost(func, prediction, target, derivative)
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

switch func
    case 'cross'
        if derivative
        else
            J = -sum(target .* log(prediction));
        end
    case 'hinge'
        if derivative
        else
            J = max(0, 1-prediction .* target);
        end
    case 'kl'
        if derivative
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

end