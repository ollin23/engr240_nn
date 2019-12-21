function [encoded] = oneHotEncoding(labelList)
% oneHotEncoding encodes the target labels using one hot encoding schema
%
% encoded = oneHotEncoding(labelList)
% For every label in a vector, labelList, each unique label is encoded
% as a one in a series of zeros the length of the number of unique labels.
% The return variable is a vector of corresponding encoded labels from the
% given list.
%
% Example -
% list = [1 2 3 4 2 1];
% encodedLabels = oneHotEncoding(list)
%       encodedLabels = [[1 0 0 0]
%                        [0 1 0 0]
%                        [0 0 1 0]
%                        [0 0 0 1]
%                        [0 1 0 0]
%                        [1 0 0 0]]

% get the unique labels
targets = unique(labelList);
% generate the one-hot matrix
targetMatrix = eye(length(targets));
[r,~] = size(labelList);

% allocate memory for encoded structure
encoded = zeros(r,length(targets));

% encode each label; 0 takes the 10th digit
for i = 1:r
    if labelList(i) == 0
        encoded(i,:) = targetMatrix(end,:);
    else
        encoded(i,:) = targetMatrix(labelList(i),:);
    end
end

end