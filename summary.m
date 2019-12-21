function summary(self, ep)


    % * * * * * * * * * * *
    %   Display Results
    % * * * * * * * * * * *
    set(gcf, 'Position', [50 30 700 500]);

    % only look at the most recent training session
    if ep ~= self.epochs || length(self.error.training) ~= ep
        errSpan = (1+(length(self.error.training)-ep)):length(self.error.training);
    else
        errSpan = 1:ep;
    end

    y1 = self.error.training(errSpan);
    
    
    if ep ~= self.epochs || length(self.stats.accuracy) ~= ep
        statsSpan = (1+(length(self.stats.accuracy)-ep)):length(self.stats.accuracy);
    else
        statsSpan = 1:ep;
    end

    
    y2 = self.stats.accuracy(statsSpan);
    y3 = self.stats.precision(statsSpan);
    y4 = self.stats.recall(statsSpan);

    
    if nnz(self.error.validation) > 0
        valErr = self.error.validation(errSpan);
    end
    if nnz(self.error.test) >0
        tstErr = self.error.test(errSpan);
    end
    

    subplot(2,2,1)
    if nnz(self.error.validation)>0 && nnz(self.error.test)>0
         plot(errSpan,y1,errSpan,valErr,errSpan,tstErr),...
            legend('Training','Validation','Test');
    elseif nnz(self.error.validation)>0
        plot(errSpan,y1,errSpan,valErr),...
            legend('Training','Validation');
    elseif nnz(self.error.test)>0
        plot(errSpan,y1,errSpan,tstErr),...
            legend('Training','Test');
    else
        plot(errSpan,y1),...
            legend('Training');
    end

    xlabel('Epochs'),ylabel('Error');
    title([self.costFunction,' Error Over ',num2str(ep),' Epochs']);
    
    subplot(2,2,2)
    plot(statsSpan,y2),xlabel('Epochs'),ylabel('Accuracy');
    title(['Accuracy Over ',num2str(ep),' Epochs']);
       
    subplot(2,2,3)
    plot(statsSpan,y3),xlabel('Epochs'),ylabel('Precision');
    title(['Precision Over ',num2str(ep),' Epochs']);
    
    subplot(2,2,4)
    plot(statsSpan,y4),xlabel('Epochs'),ylabel('Recall');
    title(['Recall Over ',num2str(ep),' Epochs']);

end