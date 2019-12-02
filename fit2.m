function fit2(self)
%fit2 executes the training cycle
%
%fit2 takes only the network as the parameter self
%
    ep = 0;
    %x = 1:self.epochs;
    epochTime = 0;

    % * * * * * * * * * * * * * * * * * *
    %   Begin Training Cycles/Epochs
    % * * * * * * * * * * * * * * * * * *
    maxEpochs = self.epochs;
    for epoch = 1:maxEpochs
        % reset longmemory
        %self.longmemory = {};        
        
        % reset time keepers
        self.stop = 0;

        % keep track of epochs
        ep = ep + 1;
        
        % continuous update on epoch percent complete
        fprintf('Epoch %d:         ',epoch);
        
        % start timer
        tic;
        % train the network
        if self.optim.parallel || self.optim.GPU
            [err, acc, prec, recall, R2] = accelTrain(self);
        else
            [err, acc, prec, recall, R2] = train2(self);
        end
        self.stop = toc;

        
        epochSummary(err, acc, prec, R2);

        epochTime = epochTime + self.stop;
        maxAcc = max(self.accuracy);
        maxPrec = max(self.precision);
        maxRecall = max(self.recall);
        maxR2 = max(self.R2);
        minError = min(self.errors);
        
        % early stop if accuracy slips past the threshold of 
        % achieved stats of the training cycle
        if self.optim.early
            threshold = .95;
            if acc <= (maxAcc * threshold)
                break;
            end
            if R2 <= (maxR2 * threshold)
                break;
            end
            if err >= (minError * threshold)
                break;
            end
        end
        
        % periodic backup
        if length(self.images.training) >= 10000
            if mod(ep,10) == 0
                trainingSummary(self, epochTime,ep,true);
            end
        else
            if mod(ep,20) == 0
                trainingSummary(self,epochTime,ep,true);
            end
        end
        
        % early stop if there over 5% of total epochs, there is more than a
        % 2% change in error in the wrong direction
        %threshold = self.epochs * .05;
        if self.optim.early && epoch >= 15
            %errAvg = sum(self.errors(end-threshold:end))/threshold;
            if err > 1.02*min(self.errors)
                break;
            end
        end
     end % end epoch

    % end of trial training summary
    trainingSummary(self, epochTime, ep, false);
    
    function epochSummary(err, acc, prec, R2)
        fprintf('\tError :\t\t%8.5f\n',err);
        fprintf('\tAccuracy :\t%8.5f\n',acc);
        fprintf('\tPrecision :\t%8.5f\n',prec);
        fprintf('\tRecall :\t%8.5f\n',recall);
        fprintf('\tR-squared :\t%8.5f\n',R2);
        fprintf('\tTime Elapsed:\t%8.5f\n',self.stop);
    end
end