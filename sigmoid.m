function[a] = sigmoid(h)
%SIGMOID takes vector h and transforms it into a sigmoid in range (0,1) 
    a = 1 / (1 + exp(-h));
end