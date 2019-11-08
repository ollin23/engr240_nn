function [] = fit(net)
%fit executes the training cycle
%
%PARAMETERS
% cycles - the nuber of epochs
    x = 1:net.epochs;
    y1 = [];
    y2 = []
    for epoch = [x];
        train(net);
        err = net.errors(end,2);
        loss = net.errors(end,1)
        fprintf('Epoch %d : Error %0.10f\n',epoch, err);
        y1 = [y1; err];
        y2 = [y2; loss];
    end
    
    plot(x,y);
end