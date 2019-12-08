function [accuracy, precision, recall, R2] = metrics(prediction, target, ybar)
% stats returns the accuracy precision recall and R-squared

    % calculate accuracy and precision
    p = round(0.5*max(0,normalize(prediction)));
        
    positives = nnz(p);
    if positives > 0
        diff = (target - p);
        FP = nnz(diff<0);
        FN = nnz(diff>0);
        TN = sum(p == diff);
        %TP = sum(p == target);
        TP = abs(1-FN);
    else
        FP = 0;
        TP = 0;
        FN = 1;
        TN = 9;
    end

    SSE = sum((target - prediction).^2);
    SST = sum((target - ybar).^2);
    
    accuracy = (TP + TN)/(FP + TP + FN + TN);
    precision = TP / (TP + FP);    
    recall = TP / (TP + FN);
    R2 = 1 - SSE/SST;


end