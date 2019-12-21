function train(net, data, labels)

if length(unique(labels)) > 2
    encoded = oneHotEncoding(labels);
else
    encoded = labels;
end
ep = 0;

for epoch = 1:net.epochs
    ep = ep + 1;
    tic;
    fprintf('Epoch %d:         ',epoch);
    err = [];
    acc = [];
    prec = [];
    rec = [];
    
    % minibatch
    if net.batchSize > 1
        batches = length(data)/net.batchSize;
    % batch
    elseif net.batchSize == 0
        batches = 1;
    % stochastic
    else
        batches = length(data);
    end
    
    start = 1;
    
    for b = 1:batches
        % minibatch
        if net.batchSize > 1
            if b > 1
                start = stop + 1;
            end
            stop = net.batchSize * b;
        % batch
        elseif net.batchSize == 0
            stop = length(data);
        % stochastic
        else
            start = b;
            stop = b;
        end
        
        X = data(start:stop,:);

        target = encoded(start:stop,:);

        prediction = net.predict(X);

        J = cost(net, prediction, target, false);
        
        %net.backprop();
        net.backprop_alt(prediction, target, X);
         
        % helper code for displaying percentage complete
        k = 2;
        if batches > 1000
            k = batches / 1000;
        end
        % dislay percentages
        if mod(b,k) == 0
            percent = 100 * (b/batches);
            strPer = ['  ' num2str(percent) '%%'];
            if percent < 100
                strCR = repmat('\b',1,length(strPer)-1);
            else
                strCR = [repmat('\b',1,length(strPer)-1) '\b\b'];
                strPer = [strPer '\n'];
            end
            fprintf([strCR strPer]);
        end
        
        
        [accuracy, precision, recall, ~] = ...
            metrics(prediction, target, mean(target));

        err = cat(2,err,mean(mean(J)));
        acc = cat(2,acc,accuracy);
        prec = cat(2,prec,precision);
        rec = cat(2,rec,recall);
        
    end
    
    net.error.training = cat(2,net.error.training, mean(err));
    net.stats.accuracy = cat(2,net.stats.accuracy, mean(acc));
    net.stats.precision = cat(2,net.stats.precision, mean(prec));
    net.stats.recall = cat(2,net.stats.recall, mean(rec));
    
    
    fprintf('\tError:\t\t%6.5f\n',net.error.training(end));
    fprintf('\tAccuracy:\t%6.5f\n',net.stats.accuracy(end));
    fprintf('\tPrecision:\t%6.5f\n',net.stats.precision(end));
    fprintf('\tRecall:\t\t%6.5f\n',net.stats.recall(end));
    endpoint = toc;
    fprintf('\tTime Elapsed:\t%0.5f\n',endpoint);

   
end

summary(net,ep);

end