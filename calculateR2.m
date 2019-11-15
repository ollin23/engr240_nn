function [R2] = calculateR2(yhat, y)
%CALCULATER2 calculates R-squared
%
% PARAMETERS
% y - target data
% yhat - prediction data

    r = y - yhat;
    normr = norm(r);
    SSE = normr.^2;
    SST = norm(y-mean(y)).^2;
    R2 = 1 - SSE/SST;

end