classdef network3 < handle
    
    properties
        layers;
        layerCount;
    end
    
    methods
        function self = net_new()
            self.layers = struct;
            self.layerCount = 0;
        end
        
        % add hidden layer
        function add(self, type, in, out, trans)
            switch nargin
                case 5
                    self.layerCount = self.layerCount + 1;
                    switch type
                        case 'linear'
                            name = [type num2str(self.layerCount)];
                            self.hidden.(name) = ...
                                struct('weights', [out in],...
                                       'transfer', trans);
                        case 'conv'
                        otherwise
                            disp('that network is not supported atm');
                    end
                otherwise
                    disp('ERROR: no layers added');
            end
        end % end add function
    end
end
