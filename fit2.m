function [] = fit2(self)
%fit executes the training cycle
%
    ep = 0;
    %x = 1:self.epochs;
    epochTime = 0;

    % * * * * * * * * * * * * * * * * * *
    %   Begin Training Cycles/Epochs
    % * * * * * * * * * * * * * * * * * *
    for epoch = 1:self.epochs
        
        ep = ep + 1;
        
        % continuous update on epoch percent complete
        fprintf('Epoch %d:         ',epoch);
        
        % start timer
        tic;
        % train the network
        [err, acc, prec, R2] = train2(self);
        t = toc;
        
%         % calculate rolling R2
%         yhat = cell2mat(self.longmemory(:,1));
%         tgt = cell2mat(self.longmemory(:,2));
%         tempR2 = cat(1,calculateR2(yhat,tgt),tempR2);
        
        fprintf('\tError :       %0.5f\n',err);
        fprintf('\tAccuracy :    %0.5f\n',acc);
        fprintf('\tPrecision :   %0.5f\n',prec);
        fprintf('\tR-squared :   %0.5f\n',R2);
        
        fprintf('\tTime Elapsed: %0.5f\n',t);

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

    end % end epoch
    
%     % calculate R2
%     self.r2 = tempR2(1);

    % * * * * * * * * * * *
    %   Display Results
    % * * * * * * * * * * *
    set(gcf, 'Position', [50 30 700 500]);
    y1 = flip(self.errors);
    y2 = flip(self.R2);
    y3 = flip(self.accuracy);
    y4 = flip(self.precision);
    x = 1:ep;
  
    subplot(2,2,1)
    plot(x,y1),xlabel('Epochs'),ylabel('Error');
    title([self.costFunction,' Error Over ',num2str(ep),' Epochs']);
    
    subplot(2,2,2)
    plot(x,y2),xlabel('Epochs'),ylabel('R^2');
    title(['R^2 Over ',num2str(ep),' Epochs']);

    subplot(2,2,3)
    plot(x,y3),xlabel('Epochs'),ylabel('Accuracy');
    title(['Accuracy Over ',num2str(ep),' Epochs']);
    
    subplot(2,2,4)
    plot(x,y4),xlabel('Epochs'),ylabel('Precision');
    title(['Precision Over ',num2str(ep),' Epochs']);
    
    % create a report
    self.timedReport(epochTime, ep);

    % reset longmemory
    self.longmemory = {};
end