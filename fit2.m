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
    for epoch = 1:self.epochs
        % reset longmemory
        self.longmemory = {};        
        
        % reset time keepers
        self.start = 0;
        self.stop = 0;

        % keep track of epochs
        ep = ep + 1;
        
        % continuous update on epoch percent complete
        fprintf('Epoch %d:         ',epoch);
        
        % start timer
        self.start = tic;
        % train the network
        [err, acc, prec, R2] = train2(self);
        self.stop = toc;
        
        epochSummary(err, acc, prec, R2);

        epochTime = epochTime + t;
        maxAcc = max(self.accuracy);
        maxPrec = max(self.precision);
        maxR2 = max(self.R2);
        minError = min(self.errors);
        
        % early stop if accuracy slips past the threshold of 
        % achieved stats of the training cycle
        if self.optim.early && epoch >= 100
            threshold = .95;
            if acc <= (maxAcc * threshold)
                break
            end
            if   R2 <= (maxR2 * threshold)
                break
            end
            if minError >= (minError * threshold)
                break;
            end
        end
        
        % early stop if there over 5% of total epochs, there is less than a
        % 2% change in major stats
        threshold = self.epochs * .05;
        if self.optim.early && epoch >= 100
            accAvg = sum(self.accuracy(end-threshold:end))/threshold;
            if abs(acc - accAvg) < .02
                break;
            end
            
        end
        
        trainSummary(self, epochTime, ep);

    end % end epoch

    function epochSummary(err, acc, prec, R2)
        fprintf('\tError :\t\t\t%8.5f\n',err);
        fprintf('\tAccuracy :\t\t%8.5f\n',acc);
        fprintf('\tPrecision :\t\t%8.5f\n',prec);
        fprintf('\tR-squared :\t\t%8.5f\n',R2);
        fprintf('\tTime Elapsed:\t%8.5f\n',self.stop);
        
    end

end