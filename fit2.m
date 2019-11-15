function [] = fit2(self)
%fit executes the training cycle
%
    ep = 0;
    x = 1:self.epochs;
    tempR2 = [];

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
        [err, acc, prec] = train2(self);
        t = toc;
        
        % calculate rolling R2
        yhat = cell2mat(self.longmemory(:,1));
        tgt = cell2mat(self.longmemory(:,2));
        tempR2 = cat(1,calculateR2(yhat,tgt),tempR2);
        
        fprintf('\tError :       %0.5f\n',err);
        fprintf('\tAccuracy :    %0.5f\n',acc);
        fprintf('\tPrecision :   %0.5f\n',prec);
        fprintf('\tR-squared :   %0.5f\n',tempR2(1));
        
        fprintf('\tTime Elapsed: %0.5f\n',t);

        epochTime = epochTime + t;
        maxAcc = max(self.accuracy);
        
        % early stop if accuracy slips past 95% of maximum achieved
        % acccuracy of the training cycle
        if self.optim.early
            if acc <= (maxAcc * .95)
                break;
            end
        end

    end % end epoch
    
    % calculate R2
    self.r2 = tempR2(1);


    % * * * * * * * * * * *
    %   Display Results
    % * * * * * * * * * * *
    set(gcf, 'Position', [50 30 700 500]);
    y1 = flip(self.errors);
    y2 = flip(tempR2);
    y3 = flip(self.accuracy);
    y4 = flip(self.precision);
  
    subplot(2,2,1)
    plot(x,y1),xlabel('Epochs'),ylabel('Error');
    title([self.costFunction,' Error Over ',num2str(self.epochs),' Epochs']);
    
    subplot(2,2,2)
    plot(x,y2),xlabel('Epochs'),ylabel('R^2');
    title(['R^2 Over ',num2str(self.epochs),' Epochs']);

    subplot(2,2,3)
    plot(x,y3),xlabel('Epochs'),ylabel('Accuracy');
    title(['Accuracy Over ',num2str(self.epochs),' Epochs']);
    
    subplot(2,2,4)
    plot(x,y4),xlabel('Epochs'),ylabel('Precision');
    title(['Precision Over ',num2str(self.epochs),' Epochs']);
    
    % create a report
    self.timedReport(epochTime, ep);

    % reset longmemory
    self.longmemory = {};
end