function [correct, wrong] = prediction(self, X, Y, labels)
%pred attempts to predict the label
%
%
%     sampleSize = length(X);
%     correctCount = 0;
%     wrongCount = 0;
    
    correct = [];
    wrong = [];
        
    for sample = 1:size(X, 1)
        a = feedforward2(self, X(sample,:));
        prediction = round(0.5*max(0,normalize(a)));
        if prediction == Y(sample,:)
            correct = cat(1,correct,prediction);
        else
            wrong = cat(1,wrong,prediction);
        end
%         if a == Y(sample,:)
%             correctCount = correctCount + 1;
%             correct = cat(2,correct,labels(sample));
%         else
%             wrongCount = wrongCount + 1;
%             wrong = cat(2,wrong,labels(sample));
%         end
    end
    
%     [m, n] = size(Y);
%     prediction = zeros(m,n);
%     
%     [L, U] = lu(prediction);
%     [m,n] = size(Y);
%     answer = zeros(m,n);
%     for i = 1:m
%         b = Y(i,:);
%         fprintf('size(b) : %d %d\n',size(b));
%         fprintf('size(L) : %d %d\n',size(L));
%         c = L \ b';
%         x = U \ c;
%         answer(m,:) = x;
%     end
%    
%     for sample = 1:sampleSize
%         
%         counter = counter + 1;
%         self.memory = {};
%         input = self.images(sample,:);
%         target = self.encodedLabels(sample,:);
%         label = self.labels(sample);
% 
%         for i = 1:length(self.weights)
%             h = feedforward2(self, input);
%             a = activate(h, self.transfer, false);
%             input = a;
%         end
% 
%         pLabel = normalize(a);
%         positives = nnz(pLabel);
%         if positives > 0
%             p = mod(find(pLabel>0),length(target));
%         else
%             p = -1;
%         end
%         
%         if p==label
%             correct = cat(1,[sample pLabel],correct);
%             correctCount = correctCount + 1;
%         else
%             wrong = cat(1,[sample pLabel],wrong);
%             wrongCount = wrongCount + 1;
%         end
%     end
end