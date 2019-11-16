function [trainSet, validateSet, testSet] = split(data, trainSize, valSize, testSize)
%split separates the data into training, validation, and test sets
%The split function splits the data into at most three separate
%non-overlapping sets
%
%EXAMPLE:
%[tr, val, tst] = split(data, .7, .2, .1);
%
%PARAMETERS
% data : the dataset to split
%
%OUTPUT
%  train : proportion of input data separated for training
%  validate : proportion of input data separated for validate
%  test : proportion of input data separated for test
%

if round(trainSize + valSize + testSize,10) ~= 1
    fprintf(['\nImproper usage of split()\n',...
             'Failed the following criterion:\n',...
             '\ttrainSize + valSize + testSize = 1\n\n']);
    fprintf(' <<< Press any button to continue >>>\n');
    pause();
elseif length(data) < 1
    fprintf(['\nImproper usage of split()\n',...
             'Failed the following criterion:\n',...
             '\tlength(data) < 2\n\n']);
    fprintf(' <<< Press any button to continue >>>\n');
    pause();
else
    % get scaled proportions of remaining data
    valtest = round(1 - trainSize,10);
    valPortion = valSize / valtest;
    tstPortion = round(1 - valPortion,10);
    
    % separate training data
    if trainSize > 0
        [n,~] = size(data);
        idx = 1:n;
        
        trainMask = randperm(n,round(trainSize*n));
        trainSet = data(trainMask,:);
        idx = setdiff(idx,idx(trainMask));
        data = data(idx,:);
    else
        trainSet = [];
    end
    
    % separate validation data
    if valPortion > 0 && tstPortion > 0
        [n,~] = size(data);
        idx = 1:n;
        valMask = randperm(n,round(valPortion*n));
        validateSet = data(valMask,:);
        idx = setdiff(idx,idx(valMask));
        data = data(idx,:);
    elseif valPortion > 0
        validateSet = data;
    else
        validateSet = [];
    end
    
    % separate test data
    if tstPortion > 0
        testSet = data;
    end
    
    if trainSize == 0
        trainSet = [];
    end
    if valSize == 0
        validateSet = [];
    end
    if testSize == 0
        testSet = [];
    end
end
end