classdef Network < handle
    % Network creates the neural network
    properties
        % internal parameters
        weights = {};       % matrices of weights between nodes
        memory = {};        % output from prior layer to current layer
        longmemory = {};    % keeps record of the epoch's output
        bias = {};          % bias
        oldDeltas = {};     % deltas used in last 
        
        % metric parameters
        errors = [];        % stores history of errors over the epochs
        accuracy = [];      % running accuracy
        precision = [];     % running precision
        R2 = [];            % r-squared
        
        % image parameters
        images = [];        % structure for images data
        labels = [];        % structure for image labels
        encodedLabels = []; % structure for one hot encoded image labels
        
        training = [];      % structure for training data
        tngLabels = [];
        tngEncLabels = [];
        
        val = [];           % structure for validation data
        valLabels = [];
        valEncLabels = [];
        
        test = [];          % structure for test data
        tstLabels = [];
        tstEncLabels = [];
               
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
                   'ridge',false,...    % L1 regularization
                   'lasso',false,...    % L2 regularization
                   'momentum',false,... % enables gradient momentum
                   'dropout',false,...  % enables random dropout
                   'ADAgrad', false,... % enables ADAgrad
                   'RMSprop', false,... % enables RMSprop
                   'early', false);     % enables early termination
               
        % other parameters
        fileName;           % name of file to save multiple trials
        start;              % start time of epoch; MATLAB tic
        stop;               % end time of epoch; MATLAB toc
        GPU;                % boolean; enables GPU usage
               
    end
    methods
        % constructor 
        function net = Network(layers, GPU)
            if nargin == 0
                % can change to generalize; this specific design is for
                % treatment of the MNIST handwritten numbers dataset
                layers = [784, 38, 16, 10];
                GPU = false;
            end
            net.GPU = GPU;
            [w, b] = createNetwork(layers, net.GPU);

            % internal parameters
            net.weights = w;
            net.bias = b;
            net.transfer = 'tanh';
            net.lastLayer = 'softmax';
            net.costFunction = 'cross';
            
            % allocate space for oldDeltas
            for i = 1:length(w)
                if net.GPU
                    net.oldDeltas{i} = zeros(size(w{i}),'gpuArray');
                else
                    net.oldDeltas{i} = zeros(size(w{i}));
                end
            end
            
            % metric parameters
            net.errors = [];
            net.accuracy = [];
            net.precision = [];
            net.R2 = [];
            if self.GPU
                net.errors = gpuArray(net.errors);
                net.accuracy = gpuArray(net.accuracy);
                net.precision = gpuArray(net.precision);
                net.R2 = gpuArray(net.R2);
            end
            
            % image parameters
            net.images = [];
            net.labels = [];
            net.encodedLabels = [];
            
            % hyperparameters
            net.epochs = 300;
            net.eta = .0001;
            net.lambda = .01;
            net.mu = .5;
            net.batches = 64;
            net.droprate = .8;
            net.trial = 1;
            if net.GPU
                net.epochs = gpuArray(net.epochs);
                net.eta = gpuArray(net.eta);
                net.lambda = gpuArray(net.lambda);
                net.mu = gpuArray(net.mu);
                net.batches = gpuArray(net.batches);
                net.droprate = gpuArray(net.droprate);
            end
        end

        % prediction function
%         function predict(self, X)
%             [correct, wrong] = predict(self, X);
%             correct;
%             wrong;
%         end
        
        % executes training cycles
        function fit(self)
            
            % if dropout active, create dropout mask
            if self.optim.dropout && ~self.optim.none
                for i = 1:length(self.weights)
                    self.dropmask{i} = ...
                        rand(size(self.weights{i})) < self.droprate;
                
                    % apply dropout mask scale remaining
                    % neurons proportionately to droprate
                    self.weights{i} = self.weights{i} .* self.dropmask{i};
                    self.weights{i} = self.weights{i} / self.droprate;
                    if self.GPU
                        self.weights{i} = gpuArray(self.weights{i});
                    end
                end
            end
            
            fit2(self);
        end

        function split(self, trainSize, valSize, testSize)
            [trainSet, valSet, testSet] =...
                split(self.images, trainSize, valSize, testSize);
            if self.GPU
                self.training = gpuArray(trainSet);
                self.val = gpuArray(valSet);
                self.test = gpuArray(testSet);
            else
                self.training = trainSet;
                self.val = valSet;
                self.test = testSet;
            end
        end
        
        % generate a report without time elapsed
        function report(self)
        % generate a report without the time elapsed recorded

            % the date time group (DTG) is the primary ID tag
            dtg = datestr(now,'yyyymmdd_HHMM');
            if self.trial == 1
                self.fileName = [pwd '\Documents\session_' dtg '.txt'];
                fileID = fopen(self.fileName,'w');
            else
                fileID = fopen(self.fileName,'a+');
            end

            %save neural network
            fName = self.fileName(1:end-4);
            networkName = [fName '_network_trial_' num2str(self.trial) '.mat'];
            save(networkName, 'self');
            
            fprintf(fileID,['\nRecord: session_' dtg '\n']);
            fprintf(fileID,'\n\nTrial %d\n',self.trial);

            fprintf(fileID,'Total epochs:  %d\n',self.epochs);
            fprintf(fileID,'Average accuracy: %0.5f\n',mean(self.accuracy));
            fprintf(fileID,'Average precision: %0.5f\n',mean(self.precision));
            fprintf(fileID,'R^2: %0.5f\n',mean(self.R2));
            fprintf(fileID,'Learning rate:  %d\n',self.eta);
            fprintf(fileID,'Batches:  %d\n',self.batches);
            fprintf(fileID,'NOTE: BGD = 0, SGD = 1\n');
            fprintf(fileID,'Regularization hyperparameter:  %d\n',self.lambda);
            fprintf(fileID,'Momentum:  %d\n',self.mu);
            fprintf(fileID,'Transfer function:  %d\n',self.transfer);
            fprintf(fileID,'Output function:  %d\n',self.lastLayer);
            fprintf(fileID,'Cost function:  %d\n',self.costFunction);
            if self.GPU
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
            
            if self.optim.ADAgrad
                outString = 'true';
            else
                outString = 'false';
            end
            fprintf(fileID,'\tADAgrad: %s\n',outString);
            
            if self.optim.RMSprop
                outString = 'true';
            else
                outString = 'false';
            end
            fprintf(fileID,'\tRMSprop: %s\n',outString);
           
            if self.optim.dropout
                outString = 'true';
                fprintf(fileID,'\tDropout: %d\n',outString);
                fprintf(fileID,'\t\t(Drop rate: %0.2f)\n',self.droprate);
            else
                outString = 'false';
                fprintf(fileID,'\tDropout: %d\n',outString);
            end
            
            if self.optim.early
                outString = 'true';
                fprintf(fileID,'\tEarly: %s\n',outString);
            else
                outString = 'false';
                fprintf(fileID,'\tEarly: %s\n',outString);
            end            
            
            
            fclose(fileID);

            figureName = [pwd '\images\' dtg 'trial_figure_' num2str(self.trial) '.fig'];
            savefig(figureName);

            self.trial = self.trial + 1;
            pause(1);
        end % end function: report
        
        % generate a report with elapsed time
        function timedReport(self, epochTime, ep)
        % generate a report with the time elapsed recorded

            % the date time group (DTG) is the primary ID tag
            dtg = datestr(now,'yyyymmdd_HHMM');
            if self.trial == 1
                self.fileName = [pwd '\Documents\session_' dtg '.txt'];
                fileID = fopen(self.fileName,'w');
            else
                fileID = fopen(self.fileName,'a+');
            end
            
            %save neural network
            fName = self.fileName(1:end-4);
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
            fprintf(fileID,'Average accuracy: %0.5f\n',mean(self.accuracy));
            fprintf(fileID,'Average precision: %0.5f\n',mean(self.precision));
            fprintf(fileID,'R^2: %0.5f\n',mean(self.R2));
            fprintf(fileID,'Learning rate:  %0.5f\n',self.eta);
            fprintf(fileID,'Batches:  %d\n',self.batches);
            fprintf(fileID,'\t**NOTE: BGD = 0, SGD = 1\n');
            fprintf(fileID,'Regularization hyperparameter:  %0.5f\n',self.lambda);
            fprintf(fileID,'Momentum:  %0.2f\n',self.mu);
            fprintf(fileID,'Transfer function:  %s\n',self.transfer);
            fprintf(fileID,'Output function:  %s\n',self.lastLayer);
            fprintf(fileID,'Cost function:  %s\n',self.costFunction);
            if self.GPU
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
            
            if self.optim.ADAgrad
                outString = 'true';
            else
                outString = 'false';
            end
            fprintf(fileID,'\tADAgrad: %s\n',outString);
            
            if self.optim.RMSprop
                outString = 'true';
            else
                outString = 'false';
            end
            fprintf(fileID,'\tRMSprop: %s\n',outString);
           
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

            figureName = [pwd '\images\' dtg 'trial_figure_' num2str(self.trial) '.fig'];
            savefig(figureName);

            self.trial = self.trial + 1;
        end %end function: timedReport

        % reset the network
        function reset(self)
        % reset the network weights, biases, memory, oldDelta, errors,
        % accuracy, and precision
        
            % reset weights and biases
            switch self.transfer
                case {'relu', 'leaky'}
                    for i = 1:length(self.weights)
                        self.weights{i} = randn(size(self.weights{i})) *...
                            (2/sqrt(length(self.weights{i})));
                        self.bias(i) = 1;
                    end
                case 'tanh'
                    for i = 1:length(self.weights)
                        self.weights{i} = randn(size(self.weights{i})) *...
                            (1/sqrt(length(self.weights{i})));
                        self.bias(i) = 1;
                    end
                otherwise
                    layers =  length(self.weights);
                    for i = 1:layers
                        H1 = numel(self.weights{i});
                        if i == 1
                            H2 = H1;
                        else
                            H2 = numel(self.weights{i-1});
                        end
                        b = sqrt(6) / sqrt(H1 + H2);
                        % if last layer
                        if i == layers
                            rows = length(unique(net.labels));
                        else
                            rows = length(self.weights{i+1});
                        end
                        columns = size(self.weights{i},1);
                        self.weights{i} = -b + (2*b)*rand([rows columns]);
                        self.bias(i) = 1;
                        if self.GPU
                            self.weights{i} = gpuArray(self.weights{i});
                            self.bias(i) = gpuArray(self.bias(i));
                        end
                    end
            end % end switch
            
            % reset the rest
            self.memory = {};
            self.oldDeltas = {};
            self.longmemory = {};
            self.errors = [];
            self.accuracy = [];
            self.precision = [];
            self.R2 = [];
            if self.GPU
                self.errors = gpuArray(self.errors);
                self.accuracy = gpuArray(self.accuracy);
                self.precision = gpuArray(self.precision);
                self.R2 = gpuArray(self.R2);
            end
                
        end % end function: reset
        
    end % end methods section
    
end % end classedef