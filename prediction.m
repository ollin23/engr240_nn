function [correct, wrong, err] = prediction(model, X, Y, labels)
%pred attempts to predict the label
%
%
    
    correct = [];
    wrong = [];
    err = [];
        
    for sample = 1:size(X, 1)
        target = Y(sample,:);
        a = feedforward2(model, X(sample,:));
        prediction = round(0.5*max(0,normalize(a)));
        if prediction == target
            correct = cat(1,labels(sample),correct);
        else
            wrong = cat(1,labels(sample),wrong);
        end

        J = cost(model, a, target, false);
        err = cat(1,J,err);
    end
    err = mean(err);
    
end