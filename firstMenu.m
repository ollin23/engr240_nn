function [datadir] = firstMenu()
% firstMenu displays the first menu of the project.
%
% OUTPUT
% datadir : the location of the file chosen for use during this session

fprintf(['\n* * * * * * * * * * * * * * * * * * * * *\n',...
           '           PROJECT BAUDRILLARD           \n',...
           '* * * * * * * * * * * * * * * * * * * * *\n']);

finished = false;
while (finished == false)

    fprintf(['\nMake a selection from following -\n',...
             '\t(1) Training set (60k images)\n',...
             '\t(2) Test set (10k images)\n',...
             '\t(3) Sample set (1000 images)\n',...
             '\t(4) Project intro\n',...
             '\t(0) Exit Program\n']);
    raw_answer = input('>>> Enter your selection:  ','s');
    if length(raw_answer) < 1
        raw_answer = '0';
    end

    answer = str2double(raw_answer);
    if length(answer) > 1
        reply = answer(1);
    else
        reply = answer;
    end

    switch reply
        case 1
            tmpFile = 'train_data.csv';
            finished = true;

        case 2
            tmpFile = 'test_data.csv';
            finished = true;

        case 3
            tmpFile = 'samples1k.csv';            
            finished = true;

        case 4
            intro();
            fprintf('\n\n\n');
        case 0
            fprintf('\nExiting the program will exit MATLAB.\n');
            raw_answer = input('Are you sure you want to exit?  (y/N) ','s');
            if length(raw_answer) < 1
                answer = 'n';
            else
                answer = raw_answer;
            end
            if strcmpi(answer,'y')
                exit;
            end
        otherwise
    end
end

    fprintf('\n%s has been selected.\n',tmpFile);
    fprintf('<<< Press any key to continue >>>\n');
    pause();

    if ispc
        datadir = [pwd '\project\' tmpFile];
    else
        datadir = [pwd '/project/' tmpFile];
    end

    % detect sample1k data file, create it if it doesn't exist
    if isfile(datadir)
        return;
    else
        tgtName = [pwd '/project/samples1k.csv'];
        
        if ispc
            srcName = [pwd '\project\test_data.csv'];
        else
            srcName = [pwd '/project/test_data.csv'];
        end
        srcData = load(srcName);
        idx = randperm(length(srcData), 1000);
        data = srcData(idx,:);
        writematrix(data,tgtName);
    end
    
    
    function intro
      fprintf(['\nThis project is an undergraduate consideration of\n',...
         'the Multilayer Perceptron (MLP). The MLP is graph\n',...
         'structure that takes a vectorized input and, through\n',...
         'one or more layers of linear and nonlinear trans-\n',...
         'formations, extracts features salient to the original\n',...
         'image which can be used to predict the category of\n',...
         'similar images.\n\n',...
         'The feature extraction through linear and nonlinear\n',...
         'transformations is called ''learning'' and is pivotal\n',...
         'to the entire fields of Machine Learning and Artificial\n',...
         'Intelligence. The MLP in particular consists of a series\n',...
         'of layers with a certain number of nodes. Each layer is\n',...
         'fully linked to its immediately adjacent layers by a\n',...
         'series of connections called weights. The first layer has\n',...
         'a number of nodes corresponding to the input, whereas\n',...
         'hidden nodes have an arbitrary number of nodes. Each node\n',...
         'provides an activation (or transfer) function which further\n',...
         'transforms the data stream, often nonlinearly. Finally,\n',...
         'the last layer, the output layer, has a number of nodes\n',...
         'commensurate with the number of labels or targets against\n',...
         'which the network is tested.\n\n',...
         'Learning occurs via backpropagation, wherein the MLP\n',...
         'output is scored against the anticipated output label and\n',...
         'the error function provides the trigger to a cascading\n',...
         'series of gradients toward the beginning of the MLP, all \n',...
         'aimed at reducing this error.\n\n',...
         'Various optimization techniques can be applied to increase\n',...
         'efficiency and performance, but due to the intense amount\n',...
         'of calculations performed, the process is time and resource\n',...
         'consuming even for toy problems. The MNIST dataset was\n',...
         'chosen because it is a well-exploited dataset and suffices\n',...
         'as the ''Hello World'' in the fields of machine learning,\n',...
         'artificial intelligence, and data science. For further\n',...
         'information regarding this project, please see\n',...
         '''Multilayer Perceptron for Numerical Image Recognition in MATLAB''\n',...
         'by Nicholas Greene and Cameron Abab.\n\n']);

      fprintf('<<< Press any key to continue >>>');
      pause();
    end
end