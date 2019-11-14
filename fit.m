function [] = fit(self)
%fit executes the training cycle
%
    ep = 0;
    x = 1:self.epochs;
    
    %cycle = round(self.epochs  / 10);

    %randomize images
    [r,~] = size(self.images);
    shuffledIndex = randperm(r);
    self.images(1:end,:) = self.images(shuffledIndex,:);    
    
    for epoch = 1:self.epochs
        ep = ep + 1;
       
        fprintf('Epoch %d:         ',epoch);
        
        % start timer
        tic;
        % train the network
        [err, acc, prec] = train2(self);
        
        fprintf('\tError :       %0.5f\n',err);
        fprintf('\tAccuracy :    %0.5f\n',acc);
        fprintf('\tPrecision :   %0.5f\n',prec);
        fprintf('\tTime Elapsed: %0.5f\n',toc);

    end
   
    set(gcf, 'Position', [100 100  400 800]);
    y1 = flip(self.errors);
    y2 = flip(self.accuracy);
    y3 = flip(self.precision);
  
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