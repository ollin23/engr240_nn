function [] = fit(net)
%fit executes the training cycle
%

    cycle = round(floor(.1* net.epochs),-1);
    ep = 0;
    x = 1:net.epochs;
    y1 = [];
    y2 = [];
    
    for epoch = x
        ep = ep + 1;
        train(net);
        err = net.errors(end,2);
        loss = net.errors(end,1);
        
        fprintf('Epoch %d : Error %0.10f\n',epoch, err);
%         % print after each 10% of epochs finish
%         if epoch == 1
%             fprintf('Epoch %d : Error %0.10f\n',epoch, err);
%         elseif (mod(epoch,cycle) == 0)
%             fprintf('Epoch %d : Error %0.10f\n',epoch, err);
%         elseif (epoch == net.epochs)
%             fprintf('Epoch %d : Error %0.10f\n',epoch, err);
%         end
        y1 = [y1; err];
        y2 = [y2; loss];
        if err <= 0
            break;
        end

    end
    
    subplot(2,1,1)
    plot(x,y1),xlabel('Epochs'),ylabel('Error');
    title(['Error Over ',num2str(ep),' Epochs']);
    
    subplot(2,1,2)
    plot(x,y2),xlabel('Epochs'),ylabel('Loss');
    title(['Loss Over ',num2str(ep),' Epochs']);
end