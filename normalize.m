function [n] = normalize(data)
%normalize normalizes data

    n = (data - min(data))/(max(data) - min(data));

end