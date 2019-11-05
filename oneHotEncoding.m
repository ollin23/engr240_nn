function [encoded] = oneHotEncoding(labelList)
% oneHotEncoding encodes the target labels using one hot encoding schema
%
% For every label in the label set, each unique label is encoded as a one
% in a series of zeros the length of the number of unique labels.
%
% Example -
% list := unique(labels) = [1 2 3 4 2 1]
% oneHotEncoding(list) => 
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

% encode each label; 0 takes the 10th digit
for i = 1:length(labelList)
    if labelList(i) == 0
        encoded(i,:) = targetMatrix(10,:);
    else
        encoded(i,:) = targetMatrix(labelList(i),:);
    end
end

end