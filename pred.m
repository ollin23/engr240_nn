function [correct, wrong] = pred(self, X)
%pred attempts to predict the label
%
%
    sampleSize = length(X);
    correctCount = 0;
    wrongCount = 0;
    
    for sample = 1:sampleSize
        
        counter = counter + 1;
        self.memory = {};
        input = self.images(sample,:);
        target = self.encodedLabels(sample,:);
        label = self.labels(sample);

        for i = 1:length(self.weightts)
            h = feedforward2(self, input);
            a = activate(h, self.transfer, false);
            input = a;
        end

        pLabel = normalize(a);
        positives = nnz(pLabel);
        if positives > 0
            p = mod(find(pLabel>0),length(target));
        else
            p = -1;
        end
        
        if p==label
            correct = cat(1,[sample pLabel],correct);
            correctCount = correctCount + 1;
        else
            wrong = cat(1,[sample pLabel],wrong);
            wrongCount = wrongCount + 1;
        end
    end
end