function [] = menuHyper(self)
%menuHyper lets the user change hyperparameter options
%
%

fprintf('\n* * * * * * * * * * * * * * *\n');
fprintf(  '*  Hyper Parameter Tuning   *');
fprintf('\n* * * * * * * * * * * * * * *\n');

finished = false;
while (finished == false)
    fprintf('\nMake a selection from the list below \n');
    fprintf(['\t(1) Number of Epochs\n',...
             '\t(2) Number of Batches\n',...
             '\t(3) Transfer function\n',...
             '\t(4) Output function\n',...
             '\t(5) Cost function selection\n',...
             '\t(6) Display current configuration\n',...
             '\t(0) Exit\n']);
    raw_answer = input(' >>> ','s');
    if length(raw_answer) < 1
        raw_answer = '0';
    end
    
    answer = str2double(raw_answer);
    if length(answer) > 1
        reply = answer(1);
    else
        reply = answer;
    end
    
    % submenu selection
    switch reply
        % epoch selection
        case 1
            hyperDisplay(answer);
            response = input('\nChange this parameter? (y/N) ','s');
            if strcmpi(response,'y')
                raw_answer = input('\nEnter new value (must be integer over zero): ','s');
                raw_answer = str2double(raw_answer);
                raw_answer = raw_answer(1);
                epochs = abs(raw_answer);
                if epochs == 0
                    epochs = 1;
                end
                self.epochs = epochs;
            end
            fprintf('\nThe network will execute %d training cycles.\n',self.epochs);
            fprintf('<< Press any key to continue >>\n');
            pause();
            
        % batches selection
        case 2
            hyperDisplay(answer);
            response = input('\nChange this parameter? (y/N) ','s');
            if strcmpi(response,'y')
                raw_answer = input('\nEnter new value (must be integer over zero): ','s');
                raw_answer = str2double(raw_answer);
                batchCount = abs(raw_answer);
                if batchCount == 0
                    batchCount = 1000;
                end
                self.batches = batchCount;
            end
            fprintf('\nEach epoch will run %d batch(es).\n',self.batches);
            fprintf('<< Press any key to continue >>');
            pause();
            
        % transfer function selection
        case 3
            transferred = false;
            while transferred == false
                hyperDisplay(answer);
                selection = ['relu', 'leaky', 'tanh', 'sigmoid'];
                response = input('\nChange this parameter? (y/N) ','s');
                if strcmpi(response,'y')
                    raw_answer = input('\nEnter new function (enter for default): ','s');
                    if strcmpi(raw_answer,'')
                        transferred = true;
                    elseif contains(selection,lower(raw_answer))
                        transferred = true;
                        self.transfer = raw_answer;
                    else
                        fprintf('\nPlease enter one of the above options.\n')
                        fprintf('<< Press any key to continue >>\n');
                        pause();
                    end
                else
                    transferred = true;
                end
            end
            fprintf('\n%s is the transfer function.\n',self.transfer);
            fprintf('<< Press any key to continue >>\n');
            pause();
            
        % output function selection    
        case 4
            doneYet = false;
            while doneYet == false
                hyperDisplay(answer);
                selection =['sigmoid', 'softmax'];
                response = input('\nChange this parameter? (y/N) ','s');
                if strcmpi(response,'y')
                    raw_answer = input('\nEnter new function (enter for default): ');
                    if strcmpi(raw_answer,'')
                        doneYet = true;
                    elseif contains(selection,lower(raw_answer))
                        doneYet = true;
                        self.lastLayer = raw_answer;
                    else
                        fprintf('\nPlease enter one of the above options.\n')
                        fprintf('<< Press any key to continue >>\n');
                        pause();
                    end
                else
                    doneYet = true;
                end
            end
            fprintf('\n%s is the output layer function.\n',self.lastLayer);
            fprintf('<< Press any key to continue >>\n');
            pause();        
                       
            
        % optimization selection
        case 5
              doneYet = false;
            while doneYet == false
                hyperDisplay(answer);
                selection = ['cross', 'hinge', 'KL', 'MSE'];
                response = input('\nChange this parameter? (y/N)','s');
                if strcmpi(response,'y')
                    raw_answer = input('\nEnter new function (enter for default): ','s');
                    if strcmpi(raw_answer,'')
                        doneYet = true;
                    elseif contains(lower(selection),lower(raw_answer))
                        doneYet = true;
                        self.costFunction = raw_answer;
                    else
                        fprintf('\nPlease enter one of the above options.\n')
                        fprintf('<< Press any key to continue >>\n');
                        pause();
                    end
                else
                    doneYet = true;                    
                end
            end
            fprintf('\nThe network will use %s for the cost.\n',self.costFunction);
            fprintf('<< Press any key to continue >>\n');
            pause();
            
        case 6
            hyperDisplay(answer);
            
        % exit
        case 0
            fprintf('\n*** Network hyperparamter setup ***\n');
            fprintf('The network will execute %d training cycles.\n',self.epochs);
            fprintf('Each epoch will run %d batch(es).\n',self.batches);
            fprintf('%s is the transfer function.\n',self.transfer);
            fprintf('%s is the output layer function.\n',self.lastLayer);
            fprintf('The network will use %s for the cost.\n',self.costFunction);
            finished = true;
        otherwise
            fprintf(['\nPlease enter a choice (0 - 7)\n',...
                    '<<Press any button to continue>>\n\n']);
            pause();
    end
end

    function hyperDisplay(choice)
        fprintf('\n');
        switch choice
            case 1
            fprintf('*** Number of Epochs Selection ***\n')
            fprintf(['\tOne epoch is one full training cycle through the\n',...
                     'network. More epochs tend to increase accuracy,\n',...
                     'to a point, but at the expense of compute time.\n\n']);
            fprintf('Current epochs = %d\t\n',self.epochs);
            
            case 2
            fprintf('*** Number of Batches Selection ***\n')
            fprintf(['The batches hyperparameter divides the sample\n',...
                     'size for each calculation into a chosen \n',...
                     'number of batches to process per epoch.\n\n',...
                     '(0) Batch Gradient Descent (BGD). The gradient\n',...
                     '\tloss are calculated after each sample but only\n',...
                     '\tapplied after each epoch.\n',...
                     '(1) Stochastic Gradient Descent (SGD). When the\n',...
                     '\tbatch size equals the sample size and the loss\n',...
                     '\tand gradient are calculated and applied after\n',...
                     '\teach training sample.\n',...
                     '(2 to %d) MiniBatch (MBGD). Any value between \n',...
                     '\tSGD and BGD. The gradient and loss are caculated\n',...
                     '\tat the end of each sample but applied after each\n',...
                     '\tbatch.\n\n'],length(self.labels));
            fprintf('Current batches = %d\t\n',self.batches);
            
            case 3
            fprintf('*** Transfer Function ***\n')
            fprintf([' ''relu'' : Rectified Linear Unit (also known as\n',...
                     '\tReLU). Returns the max of the input linear\n',...
                     '\tfunction or zero.\n',...
                     ' ''leaky'' : Leaky ReLU. If h > 0, returns linear\n',...
                     '\tfunction. If h < 0, returns the h * .01.\n',...
                     ' ''tanh'' : hyperbolic tangent. Transforms h to the\n',...
                     '\trange of (-1,1)\n',...
                     ' ''sigmoid'' : squeezes linear function to (0,1)\n\n']);
            fprintf('Current transfer function = %s\t\n',self.transfer);
            
            case 4
            fprintf('*** Output Function ***\n')
            fprintf([' ''softmax'' : softmax function (see softmax notes).\n'...
                     '\tUsed primarily for the output layer.\n',...
                     ' ''sigmoid'' : squeezes linear function to (0,1)\n\n']);
            fprintf('Current output function = %s\t\n',self.lastLayer);
            
            case 5
            fprintf('*** Cost function selection ***\n');
            fprintf(['''cross'' : cross entropy\n',...
                     '\tcalculates the cross-entropy of between the\n',...
                     '\tprediction data and the labeled data.\n',...
                     '''hinge'' : hinge loss\n',...
                     '\thinge loss used for classifiers; typically\n',...
                     '\tused for support vector machines (SVMs), but\n',...
                     '\tapplicable elsewhere.\n',...
                     '''KL'' : KullBack-Leibler Divergence\n',...
                     '\talso called relative entropy, KL is a measure\n',...
                     '\tof how the predicted probability distribution\n',...
                     '\tdiffers from the labeled data.\n',...
                     '''MSE'' : Mean Squared Error(MSE)\n',...
                     '\tcalculates the mean squared error the prediction\n',...
                     '\tdata and the labeled data\n\n']);
            fprintf('Current cost function = %s\t\n',self.costFunction);

            
            case 6
                fprintf('\nThe current hyperparameter configuration -\n');
                fprintf('\tepochs : %d\n',self.epochs);
                fprintf('\tbatches : %d\n',self.batches);
                fprintf('\ttransfer function : %s\n',self.transfer);
                fprintf('\toutput function : %s\n',self.lastLayer);
                fprintf('\tcost function : %s\n\n',self.costFunction);
                fprintf('<< Press any key to continue >>\n');
                pause();
        end
    end

% adjust the initialization of weights according to the transfer function
switch self.transfer
    case {'relu', 'leaky'}
        fprintf('Re-establishing network weights:\n');
        fprintf('ReLU and Leaky ReLU work best using Golorot initialization.\n');
        
        for i = 1:length(self.weights)
            self.weights{i} = randn(size(self.weights{i})) *...
                (2/sqrt(length(self.weights{i})));
        end
    case 'tanh'
        for i = 1:length(self.weights)
            fprintf('Re-establishing network weights...'\n);
            fprintf('Tanh work best using Xavier initialization.\n');
            self.weights{i} = randn(size(self.weights{i})) *...
                (1/sqrt(length(self.weights{i})));
        end
end

pause();

end % end of function
