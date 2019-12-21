function [shuffled] = shuffle(data, labels, tng, val, tst)

switch nargin
    % only data with labels passed in
    case {2 3}
        mask = randperm(size(data,1));
        shuffled.training.samples = data(mask,:);
        shuffled.training.labels = labels(mask,:);
        
    % data with labels , tng, and val
    case 4
        if tng + val == 1
            mask = randperm(size(data,1));
            data = data(mask,:);
            labels = labels(mask,:);
        
            tngSize = round(size(data,1) * tng);
            
            shuffled.training.samples = data(1:tngSize,:);
            shuffled.validation.samples = data(tngSize+1:end,:);
            
            shuffled.training.labels = labels(1:tngSize,:);
            shuffled.validation.labels = labels(tngSize+1:end,:);
        else
            disp('ERROR: numeric values must equal 1');
        end
        
    case 5
        if tng + val + tst == 1
            mask = randperm(size(data,1));
            data = data(mask,:);
            labels = labels(mask,:);
        
            tngSize = round(size(data,1) * tng);
            valSize = round(size(data,1) * val) + tngSize;
            
            shuffled.training.samples = data(1:tngSize,:);
            shuffled.validation.samples = data(tngSize+1:valSize,:);
            shuffled.test.samples = data(valSize+1:end,:);
            
            shuffled.training.labels = labels(1:tngSize,:);
            shuffled.validation.labels = labels(tngSize+1:valSize,:);
            shuffled.test.labels = labels(valSize+1:end,:);
            
            
        else
            disp('ERROR: numeric values must equal 1');
        end

    otherwise
        disp('ERROR : shuffle needs data, labels, and 0 to 3 integer values');
end

end