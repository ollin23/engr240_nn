clear;
clc;
%* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
% SECTION I: RETRIEVE AND DECOMPRESS DATA FROM INTERNET
%* * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
%
% online locations for binary data files
fprintf(['This program downloads the MNIST data and saves\n',...
         'it as a csv. If you already have the MNIST files,\n',...
         'please store them in the current directory under\n',...
         'the subdirectory \\project.\n\n']);
answer = input('Download and save MNIST data? (y/N)  ','s');
answer = lower(answer);

if strcmp(answer,'y')

    disp('Downloading MNIST data...');
    
    fileArray = {
     'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz';
     'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz';
     'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz';
     'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'};
    
    %unzip binary data file into \project subfolder in the current directory
     folder = [pwd '\project'];
    endFiles = {'train_images';
                'train_labels';
                'test_images';
                'test_labels'};
    
    % download and unzip files
    datafiles = {};
    metadata = [];
    for i = 1:size(fileArray,1)
         temp = char(gunzip(fileArray{i}, folder));
         endPoint = [folder '\' endFiles{i}];
         movefile(temp, endPoint);
        
        % open binary files. the metadata is uint32; the image and label data 
        % is uint8 
        file = fopen(endPoint,'r','b');
        filemeta{i} = fread(file,'uint32');
        fclose(file);
        
        file = fopen(endPoint,'r','b');
        datafiles{i} = fread(file, 'uint8');
            fclose(file);
        if filemeta{i}(1) == 2051
            metadata(i,:) = filemeta{i}(1:4);
            datafiles{i} = uint8(datafiles{i}(17:end));
        elseif filemeta{i}(1) == 2049
            metadata(i,:) = filemeta{i}(1:4);
            metadata(i,3:4) = [0 0];
            datafiles{i} = uint8(datafiles{i}(9:end));
        else
            disp('ERROR: The file is corrupt');
        end
    end
    
%* * * * * * * * * * * * * * * * * * * * * * *
%   SECTION II: PARSE DATA INTO DATA FILES
%* * * * * * * * * * * * * * * * * * * * * * *
    disp('Parsing data...');
    pause(.5);
    % save files as csv
    train_data = [];
    test_data = [];
    
    % calculates necessary dimensions for image vectors
    % and creates them
    for i = 1:length(metadata)
        rows = metadata(i,2);
        sz = (metadata(i,3) * metadata(i,4));
        cols = sz + 1;
        outfile = [];
        first = 1;
        last = sz;
        if metadata(i,1) == 2051 %image file identifier
            if i < 3
                fprintf(['* * * * * * * *\n'...
                         ' Training data \n'...
                         '* * * * * * * *\n']);
            else
                fprintf(['* * * * * * * *\n'...
                        '   Test data   \n'...
                        '* * * * * * * *\n']);            
            end
            fprintf('Percent complete:     ');
            headers = [datafiles{i+1}];
            for r = 1:rows
                outfile(r,:) = [datafiles{i}(first:last)];
                first = first + sz;
                last = last + sz;
                
                % keep track of percentage done
                percent = 100 * (r/rows);
                strPer = [num2str(percent) '%%'];
                if percent < 100
                    strCR = [repmat('\b',1,length(strPer)-1)];
                else
                    strCR = [repmat('\b',1,length(strPer)-1) '\b\b'];
                    strPer = [strPer '\n'];
                end
                fprintf([strCR strPer]);
            end
            
            % store in datasets in vectors to translate into csv files
            if i == 1
                train_data = [headers outfile];
            elseif i == 3
                test_data = [headers outfile];
            end        
        end
    end
    
    fprintf('Training data size: %d %d\n',size(train_data));
    fprintf('Test data size: %d %d\n',size(test_data));
    
    training = [folder '\train_data.csv'];
    testing = [folder '\test_data.csv'];
    
    % determine the version of MATLAB
    v = ver;
    for k = 1:length(v)
        if strcmp(v(k).Name,'MATLAB')
            version = str2double(v(k).Version);
            break;
        end
    end
    
    % save csv file
    if version < 9.6
        fprintf('Saving training data...\n');
        csvwrite(training,train_data);
        fprintf('Saving test data...\n');
        csvwrite(testing,test_data);
    else
        fprintf('Saving training data...\n');
        writematrix(train_data, training);
        fprintf('Saving test data...\n');
        writematrix(test_data, testing);
    end


    fprintf(['\n* * * * * * * * * * * \n',...
             'Conversion of MNIST binary to csv complete.\n',...
             'Use "ProjectMain.m" to begin the program.\n\n']);
else         
    fprintf(['* * * * * * * * * *\n',...
             'Use "ProjectMain.m" to begin the program.\n',...
             ' << Press any button to continue >>\n']);
    pause();         
end