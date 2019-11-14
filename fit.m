function [] = fit(self)
%fit executes the training cycle
%
    ep = 0;
    x = 1:self.epochs;
    
    %cycle = round(self.epochs  / 10);

    for epoch = 1:self.epochs
        ep = ep + 1;
        
%         self.errors = [];
%         self.accuracy = [];
        
        %randomize images
        [r,~] = size(self.images);
        shuffledIndex = randperm(r);
        self.images(1:end,:) = self.images(shuffledIndex,:);
        
        fprintf('Epoch %d:         ',epoch);
        
        % start timer
        tic;
        % train the network
        [err, acc, prec] =train2(self);
        
%         fprintf('\nepoch = %d\n',epoch);
%         fprintf('\nlength(self.errors) = %d\n',length(self.errors));
%         err = self.errors(epoch-1);
%         acc = self.accuracy(epoch-1);

        fprintf('\tError :       %0.5f\n',err);
        fprintf('\tAccuracy :    %0.5f\n',acc);
        fprintf('\tPrecision :   %0.5f\n',prec);
        
        fprintf('\tTime Elapsed: %0.5f\n',toc);
%         % print after each 10% of epochs finish
%         if epoch == 1
%             fprintf('Epoch %d :\n',epoch);
%             fprintf('\tError :\t\t %0.5f\n',self.errors(ep));
%             fprintf('\tAccuracy :\t  %0.5f\n',self.accuracy(ep));
%         elseif (mod(epoch,cycle) == 0)
%             fprintf('Epoch %d :\n',epoch);
%             fprintf('\tError :\t\t %0.5f\n',self.errors(ep));
%             fprintf('\tAccuracy :\t  %0.5f\n',self.accuracy(ep));
%         elseif (epoch == self.epochs)
%             fprintf('Epoch %d :\n',epoch);
%             fprintf('\tError :\t\t %0.5f\n',self.errors(ep));
%             fprintf('\tAccuracy :\t  %0.5f\n',self.accuracy(ep));
%         end

    end
    
   
    set(gcf, 'Position', [100 100  400 750]);
    y1 = flip(self.errors);
    y2 = flip(self.accuracy);
    
    
    subplot(3,1,1)
    plot(x,y1),xlabel('Epochs'),ylabel('Error');
    title([self.costFunction,' Error Over Time ',num2str(self.epochs),' Epochs']);

    subplot(3,1,2)
    plot(x,y2),xlabel('Epochs'),ylabel('Accuracy');
    title(['Accuracy Over ',num2str(self.epochs),' Epochs']);
    
    subplot(3,1,3)
    plot(x,y3),xlabel('Epochs'),ylabel('Precision');
    title(['Precision Over ',num2str(self.epochs),' Epochs']);
    
end