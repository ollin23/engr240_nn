function [c] = cost(func, prediction, actual, targetMatrix)

[cols, ~] = size(targetMatrix);

rows = int32(length(prediction)/ cols);

prediction = reshape(prediction,rows,cols);

output = [];
for i = 1:length(actual)
    if actual(i) == 0
        output(i,:) = targetMatrix(10,:);
    else
        output(i,:) = targetMatrix(actual(i),:);
    end
end

switch func
    case 'cross'
        n = length(prediction);
        
        c = -(1/n) * sum(output.*log(prediction)+...
            (1 - output).*log(1-prediction));

    case 'mse'
        c = .5 * sum(prediction - output).^2;
    otherwise
end

end