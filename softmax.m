function [a] = softmax(h)
%SOFTMAX takes vector h and transforms it into a probability in range [0,1]

    a = exp(h)/ sum(exp(h));

end