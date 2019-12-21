function [a] = softmax(h)
%SOFTMAX takes vector h and transforms it into a probability in range [0,1]
%
% variables:
% h - the input matrix; the hypothesis generated from previous layers via
%     the weights * input + bias (W'x + b)
% ex - created for numerical stabilization. This precludes overflow and
%     prevents division by zero from negatives with obscenely large 
%     exponents
% a - the activated data; used as output data to compare/constrast with
%     labeled data

    ex = exp(h - max(h));
    a = ex ./ sum(exp(ex));
    %a = exp(h) ./ sum(exp(h));

end