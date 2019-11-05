function [J] = cost(func, prediction, actual)
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
%          is a measure of how the prediction probability distribution differs
%          from the labeled data
%   default: MSE (mean squared error)
%       :: calculates the mean squared error the prediction data and
%          the labeled data

n = length(prediction);

switch func
    case 'cross'
        J = -sum(actual .* log(prediction)) / n;
    case 'hinge'
        J = max(0, 1-prediction .* actual);
    case 'KL'
        J = sum(prediction .* log(prediction ./ actual));
    otherwise % default is MSE
        J = (sum(prediction - actual).^2) / n;
end

end