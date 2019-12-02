function trainingSummary(self, epochTime, ep, backup)
    % * * * * * * * * * * *
    %   Display Results
    % * * * * * * * * * * *
    fprintf('ep : %d\n',ep);
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

    if epochTime > 0
        self.report(epochTime,ep,backup);
    end

end