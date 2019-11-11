function [] = fit(self)
%fit executes the training cycle
%

    cycle = round(floor(.1* self.epochs),-1);
    ep = 0;
    x = 1:self.epochs;
    y1 = [];
    y2 = [];
    
    for epoch = x
        ep = ep + 1;
        %train(net);
        fprintf('Epoch %d:       ',epoch);
        
        % start timer
        tic;
        train2(self);
        err = self.errors(end,2);
        loss = self.errors(end,1);
        
        fprintf('\tError %0.10f\n',err);
        fprintf('\tTime Elapsed: %0.10f\n',toc);
%         % print after each 10% of epochs finish
%         if epoch == 1
%             fprintf('Epoch %d : Error %0.10f\n',epoch, err);
%         elseif (mod(epoch,cycle) == 0)
%             fprintf('Epoch %d : Error %0.10f\n',epoch, err);
%         elseif (epoch == self.epochs)
%             fprintf('Epoch %d : Error %0.10f\n',epoch, err);
%         end
        y1 = cat(1,err,y1);
        y2 = cat(1,loss,y2);
        if err <= 0
            break;
        end

    end
    
    subplot(2,1,1)
    plot(x,y1),xlabel('Epochs'),ylabel('Error');
    title(['Accuracy ',num2str(ep),' Epochs']);
    
    subplot(2,1,2)
    plot(x,y2),xlabel('Epochs'),ylabel('Loss');
    title(['Loss Over ',num2str(ep),' Epochs']);
end