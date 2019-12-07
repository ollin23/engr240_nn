function trainingSummary(self, epochTime, ep, backup)


    % * * * * * * * * * * *
    %   Display Results
    % * * * * * * * * * * *
    fprintf('ep : %d\n',ep);
    set(gcf, 'Position', [50 30 700 500]);

    if ep ~= self.epochs
        x = (1+(length(self.errors)-ep)):length(self.errors);
    else
        x = 1:ep;
    end

    y1 = self.errors(x);
    y2 = self.R2(x);
    y3 = self.accuracy(x);
    y4 = self.precision(x);
    
    y1 = flip(y1);
    y2 = flip(y2);
    
    y3 = flip(y3);
    y4 = flip(y4);
    
    if nnz(self.error.validation) > 0
        valErr = self.error.validation(x);
        valErr = flip(valErr);
    end
    if nnz(self.error.test) >0
        tstErr = self.error.test(x);
        tstErr = flip(tstErr);
    end
    

    subplot(2,2,1)
    if nnz(self.error.validation)>0
        plot(x,y1,x,valErr),...
            legend('Training','Validation');
    elseif nnz(self.error.test)>0
        plot(x,y1,x,tstErr),...
            legend('Training','Test');
    elseif nnz(self.error.validation)>0 && nnz(net.error.test)>0
        plot(x,y1,x,valErr,x,tstErr),...
            legend('Training','Validation','Test');
    else
        plot(x,y1),...
            legend('Training');
    end

    xlabel('Epochs'),ylabel('Error');
    title([self.costFunction,' Error Over ',num2str(ep),' Epochs']);
    
    subplot(2,2,2)
    plot(x,y3),xlabel('Epochs'),ylabel('Accuracy');
    title(['Accuracy Over ',num2str(ep),' Epochs']);
       
    subplot(2,2,3)
    plot(x,y2),xlabel('Epochs'),ylabel('R^2');
    title(['R^2 Over ',num2str(ep),' Epochs']);
    
    subplot(2,2,4)
    plot(x,y4),xlabel('Epochs'),ylabel('Precision');
    title(['Precision Over ',num2str(ep),' Epochs']);

    if epochTime > 0
        self.report(epochTime,ep,backup);
    end

    
end