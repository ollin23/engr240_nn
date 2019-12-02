function splitData(self, trainSize, valSize, testSize)
%split separates the data into training, validation, and test sets
%The split function splits the data into at most three separate
%non-overlapping sets
%
%EXAMPLE:
%split(.7, .2, .1);
%
%PARAMETERS
% data : the dataset to split
%
%OUTPUT
%   training : proportion of input data separated for training
%   tngLabels : training labels
%   tngEncLabels : encoded training labels
%   val : proportion of input data separated for validation
%   valLabels : validation labels
%   valEncLabels : encoded validation labels
%   test : proportion of input data separated for test
%   tstLabels : test labels
%   tstEncLabels : encoded test labels

data = self.imageData;
labels = self.labels;
encLabels = self.encodedLabels;


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
    valtest = round(1 - trainSize,10);  % adjust for float error near 0
    valPortion = valSize / valtest;
    tstPortion = round(1 - valPortion,10);
    
    % separate training data
    if trainSize > 0
        [n,~] = size(data);
        idx = 1:n;
        
        % extract training images and labels
        trainMask = randperm(n,round(trainSize*n));
        self.images.training = data(trainMask,:);
        self.images.tngLabels = labels(trainMask);
        self.images.tngEncLabels = encLabels(trainMask,:);
        
        idx = setdiff(idx,idx(trainMask));
        data = data(idx,:);
        labels = labels(idx);
        encLabels = encLabels(idx,:);
    else
        self.images.training = [];
        self.images.tngLabels = [];
        self.images.tngEncLabels = [];
    end
    
    % separate validation data
    if valPortion > 0 && tstPortion > 0
        [n,~] = size(data);
        idx = 1:n;
        
        % extract validation images and labels
        valMask = randperm(n,round(valPortion*n));
        self.images.val = data(valMask,:);
        self.images.valLabels = labels(valMask);
        self.images.valEncLabels = encLabels(valMask,:);
        
        idx = setdiff(idx,idx(valMask));
        data = data(idx,:);
        labels = labels(idx);
        encLabels = encLabels(idx,:);
    elseif valPortion > 0
        self.images.val = data;
        self.images.valLabels = labels;
        self.images.valEncLabels = encLabels;
    else
        self.images.val = [];
        self.images.valLabels = [];
        self.images.valEncLabels = [];
    end
    
    % separate test data
    if tstPortion > 0
        self.images.test = data;
        self.images.tstLabels = labels;
        self.images.tstEncLabels = encLabels;
    end
    
    if trainSize == 0
        self.images.training = [];
        self.images.tngLabels = [];
        self.images.tngEncLabels = [];
    end
    if valSize == 0
        self.images.val = [];
        self.images.valLabels = [];
        self.images.valEncLabels = [];
    end
    if testSize == 0
        self.images.test = [];
        self.images.tstLabels = [];
        self.images.tstEncLabels = [];
    end
end
end