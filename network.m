classdef network < handle
    
    properties
        layers;
        memory;
        layerCount;
        backupLayers;
        
        epochs;         % training cycles
        eta;            % learning rate
        lambda;         % regularization factor
        mu;             % momentum factor
        
        batchSize;      % samples per batch
        costFunction = '';   % cost function
        
        error = ...
            struct('training', [],...
                   'validation', [],...
                   'test', []);
        stats = ...
            struct('accuracy', [],...
                   'precision',[],...
                   'recall',[]);
               
        options = ...
            struct('none', true,...
                   'lasso', false,...
                   'ridge', false,...
                   'momentum',false,...
                   'dropout',false,...
                   'GPU',false,...
                   'adam',false,...
                   'early',false);
    end
    
    methods
        % constructor function
        function self = network(varargin)
            rng(23,'twister');
            
            self.layers = struct;
            self.layerCount = 0;
            
            self.epochs = 200;
            self.eta = 1e-5;
            self.lambda = 1e-15;
            self.mu = .9;
            self.batchSize = 100;
            self.costFunction = 'sparse_cross';
            
            
            % create network
            nodes = [];
            if ~isempty(varargin)
                for v = 1:length(varargin)
                    if isreal(varargin{v}) && rem(varargin{v},1)==0
                        nodes = cat(2,nodes,varargin{v});
                    end
                end
            end

            if ~isempty(nodes)
                for i = 1:length(nodes)-1
                    if i < length(nodes)-1
                        self.add('dense',nodes(i),nodes(i+1),'leaky');
                    else
                        self.add('dense',nodes(i),nodes(i+1),'softmax');
                    end
                end
            end
        end % end constructor
        
        % add layers
        function add(self, type, in, out, trans)
            switch nargin
                case 5
                    self.layerCount = self.layerCount + 1;
                    switch type
                        case 'dense'
                            name = [type num2str(self.layerCount)];
                            self.layers.(name) = ...
                                struct('weights', randn([in out]),...
                                       'bias', 1,...
                                       'transfer', trans);
                        case 'conv'
                        case 'pool'
                        otherwise
                            disp('that network is not supported atm');
                    end
                otherwise
                    disp('ERROR: Layers improperly formed');
            end
            self.backupLayers = self.layers;
        end % end add function
        
        % feedforward function
        function h = predict(self, X)
            fn = fieldnames(self.layers);
            h = X;
            for k = 1:numel(fn)
                h = h * self.layers.(fn{k}).weights + ...
                       self.layers.(fn{k}).bias;
                self.memory.(fn{k}).h = h;
                h = self.activate(h,self.layers.(fn{k}).transfer, false);
            end
        end
        
        % non-linear activation function
        function a = activate(~, h, trans, derivative)
            switch trans
                case 'relu'
                    if ~derivative
                        a = h .* (h>0);
                    else
                        a = 1.* (h>0);
                    end
                case 'leaky'
                    if ~derivative
                        a = h.*(h>0) + .01*(h .* (h<0));
                    else
                        a = 1.*(h>0) + .01*(1 .* (h<0));
                    end
                case 'tanh'
                    if ~derivative
                        a = tanh(h);
                    else
                        a = sech(h).^2;
                    end
                case 'sigmoid'
                    if ~derivative
                        a = sigmoid(h);
                    else
                        a = sigmoid(h).* (1 - sigmoid(h));
                    end
                case 'softmax'
                    if ~derivative
                        a = softmax(h);
                    else
                        a = softmax(h) .* (1-softmax(h));
                        %a = gradient(a);
                    end
                otherwise
            end
        end
  
        % alternate backprop algorithm
        function backprop_alt(self, prediction, target, input)
            fn = fieldnames(self.layers);
          
            for k = numel(fn):-1:1
                                
                if k == numel(fn)
                    g = cost(self, prediction, target, true);
                    h = self.memory.(fn{k-1}).h;
                    a = self.activate(h,self.layers.(fn{k}).transfer, true);
                else
                    g = g * self.layers.(fn{k+1}).weights';
                    
                
                    if k == 1
                        a = input;
                    else
                        h = self.memory.(fn{k-1}).h;
                        a = self.activate(h,...
                            self.layers.(fn{k}).transfer,...
                            true);
                    end
                end            
                
                delta = g' * a;
                delta = self.eta * delta';
                
                % optimizations
                if ~self.options.none
                    delta = self.optimize(delta, fn, k);
                end
                
                self.memory.(fn{k}).weights = self.layers.(fn{k}).weights;
                self.memory.(fn{k}).bias = self.layers.(fn{k}).bias;
                self.memory.(fn{k}).gradient = delta;
                
%               fprintf('size(delta) : %d %d\n',size(delta));
%               fprintf('size(w) : %d %d\n',size(self.layers.(fn{k}).weights));
                self.layers.(fn{k}).weights = ...
                    self.layers.(fn{k}).weights - delta;
                self.layers.(fn{k}).bias = ...
                    self.layers.(fn{k}).bias - mean(mean(delta));
            end
        end
        
        % backpropagation
        function backprop(self)
            fn = fieldnames(self.layers);
          
            for k = numel(fn):-1:1
                if k == numel(fn)
                    h = self.memory.(fn{k-1}).h;
                    t = self.layers.(fn{k}).transfer;
                    g = gradient(self.activate(self,h,t,false));
                    if self.batchSize ~= 1
                        g = mean(g);
                    end
                else
                    h = self.memory.(fn{k}).h;
                    t = self.layers.(fn{k}).transfer;
                    g = gradient(self.activate(self,h,t,false));
                end
                
                delta = self.eta * g;
                
                if ~self.options.none
                    delta = self.optimize(delta, fn, k);
                end
             
                self.memory.(fn{k}).weights = self.layers.(fn{k}).weights;
                self.memory.(fn{k}).bias= self.layers.(fn{k}).bias;
                self.memory.(fn{k}).gradient = delta;
                
                self.layers.(fn{k}).weights = ...
                    self.layers.(fn{k}).weights - delta;
                self.layers.(fn{k}).bias = ...
                    self.layers.(fn{k}).bias - mean(mean(delta));
            end
        end
      
        % optimizations
        function [delta] = optimize(self, delta, fn, k)
            % * * * * * * * * * * * * * * * * 
            %        OPTIMIZATIONS
            % * * * * * * * * * * * * * * * * 
            if ~(self.options.none)
                if self.options.momentum
                    try
                        delta = delta + self.mu * self.memory.(fn{k}).gradient;
                    catch
                        % do nothing, there is no memory yet
                    end
                end
                if self.options.ridge
                    delta = delta + ...
                        self.lambda * ...
                        sum(sum(self.layers.(fn{k}).weights .^2));

                elseif self.options.lasso
                    delta = delta + ...
                        self.lambda *...
                        abs(sum(sum(self.layers.(fn{k}).weights)));
                end
            end
        end
        
        % reset the network
        function reset(self, variant)
            switch variant
                % clear error and stats
                case 1
                    clear self.error.training self.error.validation ...
                        self.error.test self.stats.accuracy ...
                        self.stats.precision self.stats.recall;
                    self.error.training = [];
                    self.error.validation = [];
                    self.error.test = [];
                    self.stats.accuracy = [];
                    self.stats.precision = [];
                    self.stats.recall = [];
                    disp('RESET: metrics only');
                    
                % weights only
                case 2
                    fn = fieldnames(self.layers);
                    for k = 1:numel(fn)
                        clear self.layers.(fn{k}).weights ...
                            self.layers.(fn{k}).bias;
                    end
                    self.layers = self.backupLayers;
                    disp('RESET: resetting weights and biases only');

                    
                % clear stats, reset weights
                case 3
                    clear self.error.training self.error.validation ...
                        self.error.test self.stats.accuracy ...
                        self.stats.precision self.stats.recall
                    fn = fieldnames(self.layers);
                    for k = 1:numel(fn)
                        clear self.layers.(fn{k}).weights ...
                            self.layers.(fn{k}).bias;
                    end
                    self.layers = self.backupLayers;
                    
                    self.error.training = [];
                    self.error.validation = [];
                    self.error.test = [];
                    self.stats.accuracy = [];
                    self.stats.precision = [];
                    self.stats.recall = [];
                    disp('RESET: reset entire network to default');
                    disp(' <<< press any key to continue >>> ');
                    pause();
                    self.epochs = 150;
                    self.eta = 1e-5;
                    self

                otherwise
                    disp('ERROR: invalid option selected');
            end
        end
    end
end