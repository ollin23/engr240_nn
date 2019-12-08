classdef Network < handle
    % Network creates the neural network
    properties
        % internal parameters
        weights = {};       % matrices of weights between nodes
        memory = {};        % output from prior layer to current layer
        longmemory = {};    % keeps record of the epoch's output
        bias = [];          % bias
        oldDeltas = {};     % deltas used in last iteration
        backupWeights = {}; % backup weights for reset
        backupBias = [];    % backup biases for reset
        
        % metric parameters
        error = ...
            struct('training', [],...
                   'validation', [],...
                   'test', []);
        errors = [];        % runnning error over the epochs
        validationErr = []; % validation error
        testErr = [];       % test error
        accuracy = [];      % running accuracy
        precision = [];     % running precision
        recall = [];        % running recall
        R2 = [];            % r-squared
        
        % image parameters
        imageData = [];     % structure for images data
        labels = [];        % structure for image labels
        encodedLabels = []; % structure for one hot encoded image labels
        images = ...
            struct('training', [],...       % structure for training data
                   'tngLabels', [],...
                   'tngEncLabels', [],...
                   'val', [],...            % structure for validation data
                   'valLabels', [],...
                   'valEncLabels', [],...
                   'test', [],...           % structure for test data
                   'tstLabels', [],...
                   'tstEncLabels', []);
               
        % hyperparameters
        epochs;             % number of cycles through the training data
        eta;                % learning rate
        lambda;             % regularization hyperparameter
        mu;                 % momentum hyperparameter
        droprate;           % drop rate for use with dropout
        dropmask = {};      % the mask for the dropout layers
        batches;            % number of batches for mini-batch training
        transfer = '';      % transfer function type
        lastLayer = '';     % activation function for output layer
        costFunction = '';  % type of cost function
        trial;              % ordinal number in series of trials
        % NOTES:    (1) none overrides all other optimization techniques
        %           (2)ADAgrad, and RMSprop are mutually exclusive
        optim = ...
            struct('none', true,...     % no optimizations used
                   'GPU', false,...     % use GPU acceleration
                   'parallel', false,... % use parallelization
                   'ridge',false,...    % L1 regularization
                   'lasso',false,...    % L2 regularization
                   'momentum',false,... % enables gradient momentum
                   'dropout',false,...  % enables random dropout
                   'adam', false,... % enables ADAgrad
                   'early', false);     % enables early termination
               
        % other parameters
        fileName;           % name of file to save multiple trials
        stop;               % end time of epoch; MATLAB toc
        threshold;          % early stop threshold
               
    end
    methods
        % constructor 
        function net = Network(layers)
            if nargin == 0
                % can change to generalize; this specific design is for
                % treatment of the MNIST handwritten numbers dataset
                layers = [784, 38, 16, 10];
            end
            [w, b] = createNetwork(layers);

            % internal parameters
            net.weights = w;
            net.oldDeltas = w;
            net.backupWeights = w;
            net.bias = b;
            net.backupBias = b;
            
            net.transfer = 'leaky';
            net.lastLayer = 'softmax';
            net.costFunction = 'cross';
            
            % metric parameters
            net.errors = [];
            net.accuracy = [];
            net.precision = [];
            net.recall = [];
            net.R2 = [];
            
            % image parameters
            net.images = [];
            net.labels = [];
            net.encodedLabels = [];
            
            % defauls hyperparameters
            net.epochs = 50;
            net.eta = .00001;
            net.lambda = .1;
            net.mu = .5;
            net.batches = 64;
            net.droprate = .85;
            net.trial = 1;
        end

        % prediction function
        function predict(self, option)
         %predict predicts results based on the trained model
         %
         % Example :
         %  net.predict('test')
         %
            switch nargin
                 case 2
                    switch option
                        case 'test'
                            try
                                X = self.images.test;
                                Y = self.images.tstEncLabels;
                                lbls = self.images.tstLabels;
                            catch
                                disp('No test samples available');
                                pause();
                            end
                        case 'validation'
                            try
                                X = self.images.val;
                                Y = self.images.valEncLabels;
                                lbls = self.images.valLabels;
                            catch
                                disp('No validation samples available.');
                                pause();
                            end
                        otherwise
                            disp('ERROR; enter "test" or "validation"');
                            
                    end
                 otherwise
                    X = self.images.test;
                    Y = self.images.tstEncLabels;
                    lbls = self.images.tstLabels;
                    option = 'test';
            end
            
            switch option
                case 'test'
                    [c, w, e] = prediction(self, X, Y, lbls);
                    self.error.test = cat(1,e,self.error.test);

                    proportion = length(c) / length(w);
                    fprintf('\tTest Set:\n');
                    fprintf('\t      correct:\t%d\n',length(c));
                    fprintf('\t\twrong:\t%d\n',length(w));
                    fprintf('\t\tratio:\t%0.2f\n',proportion);
                case 'validation'
                    [c, w, e] = prediction(self, X, Y, lbls);
                    self.error.validation = cat(1,e,self.error.validation);
                    
                    proportion = length(c) / length(w);
                    fprintf('\tCross Validation:\n');
                    fprintf('\t      correct:\t%d\n',length(c));
                    fprintf('\t\twrong:\t%d\n',length(w));
                    fprintf('\t\tratio:\t%0.2f\n',proportion);
                otherwise
                    disp('ERROR: no type chosen for test/validation');
            end
         end
        
        % executes training cycles
        function fit(self, enabled)
        % executes training cycles
        
            self.enableAcceleration(enabled);
            
            % if dropout active, create dropout mask
            if self.optim.dropout && (self.optim.none == false)
                for i = 1:length(self.weights)
                    self.dropmask{i} = ...
                        rand(size(self.weights{i})) < self.droprate;
                
                    % apply dropout mask scale remaining
                    % neurons proportionately to droprate
                    self.weights{i} = self.weights{i} .* self.dropmask{i};
                    self.weights{i} = self.weights{i} / self.droprate;
                    if self.optim.GPU && enabled
                        self.weights{i} = gpuArray(self.weights{i});
                    end
                end
            end
            if self.optim.parallel
                try
                    pool = gcp;
                catch
                    self.optim.parallel = false;
                    disp('MATLAB initialize parallelization at this time.');
                    pause(.1);
                    if ~isempty(pool)
                        fprintf('Parallel CPU pool already started.\n');
                    else
                        fprintf('Parallel pool is empty.\n');
                    end
                end
            end

            fit2(self);
            
            if self.optim.parallel
                delete(pool);
            end
        end
        
        % enable GPU acceleration
        function enableAcceleration(self, enable)
        %enableGPUacceleration tests GPU available and gives the user an option to
        %use it or not

            % test for GPU access
            try
                gpuArray(1);
            catch
                self.optim.GPU = false;
                disp('This system cannot use GPUs for data processing in MATLAB.');
            end

            if self.optim.GPU && (enable == false)
                fprintf('GPU usage is disabled for this session.\n\n');
                self.optim.GPU = false;
            end
             % enable GPU acceleration
             try
                if self.optim.GPU && enable
                    disp('Enabling GPU acceleration...');
                    for i = 1:length(self.weights)
                        fprintf('Creating gpuArray for weights, layer %d\n',i);
                        pause(.05);
                        self.weights{i} = gpuArray((self.weights{i}));
                    end

                    self.images.tngEncLabels = gpuArray(self.images.tngEncLabels );
                    fprintf('Creating gpuArray encoded labels\n');
                    pause(.05);

                    self.images.training = gpuArray(self.images.training);
                    self.images.tngLabels = gpuArray(self.images.tngLabels);
                end
             catch
                 disp('ERROR: unable to dispatch GPU acceleration on this system.');
             end
        end
        
        % split images into training, validation, and test sets
        function split(self, tng, vl, tst)
        % split the data into training, validation, and testing sets
        
            switch nargin
                case 1
                    trainSize = 1;
                    valSize = 0;
                    testSize = 0;
                case 2
                    trainSize = tng;
                    valSize = 0;
                    testSize = 0;
                case 3
                    trainSize = tng;
                    valSize = vl;
                    testSize = 0;
                otherwise
                    trainSize = tng;
                    valSize = vl;
                    testSize = tst;
            end
            splitData(self, trainSize, valSize, testSize);
        end

        function shuffle(self, varargin)
            vars = cellstr(varargin);

            if nargin > 1
                % test data
                if any(strcmpi(vars,'test'))
                    data = self.images.test;
                    lbls = self.images.tstLabels;
                    encoded = self.images.tstEncLabels;

                    trainMask = randperm(size(data,1));
                    data = data(trainMask,:);
                    lbls = lbls(trainMask);
                    encoded = encoded(trainMask,:);

                    self.images.test = data;
                    self.images.tstLabels = lbls;
                    self.images.tstEncLabels = encoded;
                end
                if any(strcmpi(vars,'validation'))
                    data = self.images.val;
                    lbls = self.images.valLabels;
                    encoded = self.images.valEncLabels;

                    trainMask = randperm(size(data,1));
                    data = data(trainMask,:);
                    lbls = lbls(trainMask);
                    encoded = encoded(trainMask,:);

                    self.images.val = data;
                    self.images.valLabels = lbls;
                    self.images.valEncLabels = encoded;
                end
                if any(strcmpi(vars,'training'))
                    data = self.images.training;
                    lbls = self.images.tngLabels;
                    encoded = self.images.tngEncLabels;

                    trainMask = randperm(size(data,1));
                    data = data(trainMask,:);
                    lbls = lbls(trainMask);
                    encoded = encoded(trainMask,:);

                    self.images.training = data;
                    self.images.tngLabels = lbls;
                    self.images.tngEncLabels = encoded;
                end
            end
        end
        
        % generate a report with elapsed time
        function report(self, epochTime, ep, backup)
        % generate a report with the time elapsed recorded
            if ispc
                rptFolder = [pwd '\reports'];
            else
                rptFolder = [pwd '/reports'];
            end
            if ~exist(rptFolder, 'dir')
                mkdir('reports');
            end
            % the date time group (DTG) is the primary ID tag
            dtg = datestr(now,'yyyymmdd_HHMM');
            if self.trial == 1
                if ispc
                    self.fileName = [pwd '\reports\session_' dtg '.txt'];
                else
                    self.fileName = [pwd '/reports/session_' dtg '.txt'];
                end
                fileID = fopen(self.fileName,'w');
            else
                fileID = fopen(self.fileName,'a+');
            end
            
            %save neural network
            fName = self.fileName(1:end-4);
            
            % save backups
            if backup
                fName = [fName '_BACKUP_epoch_' num2str(ep)];
            end

            networkName = [fName '_network_trial_' num2str(self.trial) '.mat'];
            
            class(networkName)
            fprintf('%s\n',networkName);
            
            save(networkName, 'self');
            
            fprintf(fileID,['\nRecord: session_' dtg '\n']);
            fprintf(fileID,'\nTrial %d\n',self.trial);

            fprintf(fileID,'Total epochs:  %d\n',self.epochs);
            if self.optim.early
                fprintf(fileID,'Epochs trained: %d\n',ep);
                fprintf(fileID,'Percent epochs completed: %0.2f\n',(ep/self.epochs));
            end
            fprintf(fileID,'Total time elapsed: %0.5f seconds\n',epochTime);
            fprintf(fileID,'Average time per epoch: %0.5f seconds\n',epochTime/self.epochs);
            fprintf(fileID,'Average error: %0.5f\n',mean(self.errors));
            fprintf(fileID,'Final error: %0.5f\n',self.errors(1));
            fprintf(fileID,'Validation error: %0.5f\n',mean(self.error.validation));
            fprintf(fileID,'Average accuracy: %0.5f\n',mean(self.accuracy));
            fprintf(fileID,'Average precision: %0.5f\n',mean(self.precision));
            fprintf(fileID,'Average recall: %0.5f\n',mean(self.recall));
            fprintf(fileID,'R^2: %0.5f\n',mean(self.R2));
            fprintf(fileID,'Learning rate:  %0.5f\n',self.eta);
            fprintf(fileID,'Batches:  %d\n',self.batches);
            fprintf(fileID,'\t**NOTE: BGD = 0, SGD = 1\n');
            fprintf(fileID,'Regularization hyperparameter:  %0.5f\n',self.lambda);
            fprintf(fileID,'Momentum:  %0.2f\n',self.mu);
            fprintf(fileID,'Transfer function:  %s\n',self.transfer);
            fprintf(fileID,'Output function:  %s\n',self.lastLayer);
            fprintf(fileID,'Cost function:  %s\n',self.costFunction);
            if self.optim.parallel
                outString = 'true';
            else
                outString = 'false';
            end
            fprintf(fileID,'Paralellization: %s\n',outString);
            
            if self.optim.GPU
                outString = 'true';
            else
                outString = 'false';
            end
            fprintf(fileID,'GPU acceleration: %s\n',outString);
            fprintf(fileID,'* Optimization schema *\n');
            if self.optim.none
                outString = 'true';
            else
                outString = 'false';
            end
            fprintf(fileID,'\tNone: %s\n',outString);
            
            if self.optim.lasso
                outString = 'true';
            else
                outString = 'false';
            end
            fprintf(fileID,'\tLasso regularization: %s\n',outString);
            
            if self.optim.ridge
                outString = 'true';
            else
                outString = 'false';
            end
            fprintf(fileID,'\tRidge regularization: %s\n',outString);
            
            if self.optim.momentum
                outString = 'true';
            else
                outString = 'false';
            end
            fprintf(fileID,'\tMomentum: %s\n',outString);
            
            if self.optim.adam
                outString = 'true';
            else
                outString = 'false';
            end
            fprintf(fileID,'\tadam: %s\n',outString);
           
            if self.optim.dropout
                outString = 'true';
                fprintf(fileID,'\tDropout: %s\n',outString);
                fprintf(fileID,'\t\t(Drop rate: %0.2f)\n',self.droprate);
            else
                outString = 'false';
                fprintf(fileID,'\tDropout: %s\n',outString);
            end
            
            if self.optim.early
                outString = 'true';
                fprintf(fileID,'\tEarly: %s\n',outString);
            else
                outString = 'false';
                fprintf(fileID,'\tEarly: %s\n',outString);
            end
            
            fclose(fileID);

            % save figure
            if ~exist('images','dir')
                mkdir('images');
            end
            if ispc
                figureName = [pwd '\images\' dtg 'trial_figure_' num2str(self.trial) '.fig'];
            else
                figureName = [pwd '/images/' dtg 'trial_figure_' num2str(self.trial) '.fig'];
            end
            savefig(figureName);

            self.trial = self.trial + 1;
        end %end function: timedReport

        % reset the network
        function reset(self, change)
        % reset the network weights, biases, memory, oldDelta, errors,
        % accuracy, and precision
        
        switch nargin
            case 2
                if (change == true)
                    switch self.transfer
                        case {'relu', 'leaky'}
                            fprintf('Re-establishing network weights:\n');
                            fprintf('ReLU and Leaky ReLU work best using Golorot initialization.\n');

                            for i = 1:length(self.weights)
                                self.weights{i} = randn(size(self.weights{i})) *...
                                    (2/sqrt(length(self.weights{i})));
                            end
                        case 'tanh'
                            % works best with tanh
                            for i = 1:length(self.weights)
                                fprintf('Re-establishing network weights...'\n);
                                fprintf('Tanh work best using Xavier initialization.\n');
                                self.weights{i} = randn(size(self.weights{i})) *...
                                    (1/sqrt(length(self.weights{i})));
                            end
                        otherwise
                            layers = [];
                            for i = 1:length(self.weights)
                                [out, in] = size(self.weights{i});
                                if i == 1
                                    layers = [in out];
                                else
                                    layers = cat(2,layers,out);
                                end
                            end
                            for i = 1:(length(layers)-1)
                                H1 = double(layers(i));
                                if i == 1
                                    H2 = double(H1);
                                else
                                    H2 = double(layers(i-1));
                                end
                                b = sqrt(6.0) / sqrt(H1 + H2);
                                self.weights{i} = -b + (2*b)*rand([layers(i+1) layers(i)]);
                                self.bias(i) = 1;
                            end
                    end
                else
                    self.weights = self.backupWeights;
                    self.bias = self.backupBias;
                end
            otherwise
                self.weights = self.backupWeights;
                self.bias = self.backupBias;
        end
            % reset the rest
            self.resetMetrics();
        end % end function: reset
        
        function resetMetrics(self)
            self.memory = {};
            self.oldDeltas = self.weights;
            self.errors = [];
            self.accuracy = [];
            self.recall = [];
            self.precision = [];
            self.R2 = [];
            
            self.error.training = [];
            self.error.validation = [];
            self.error.test = [];
        end
        
    end % end methods section
    
end % end classedef