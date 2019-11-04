function [nodesPerLayer] = menu()
%menu creates a menu to select architecture of the network
%
% menu() walks the user through individual choices to design a network
% topology. 
nodesPerLayer = [];

answer = -1;

fprintf('\n* * * * * * * * * * * * * * *');
fprintf('\n    Building the topology    ');
fprintf('\n* * * * * * * * * * * * * * *\n');

% set max number of layers and nodes
maxLayers = 5;
maxNodes = 300;

% get the number of layers
while (answer < 0)
    fprintf('<<Enter 0 for the default>>\n');
    fprintf('How many hidden layers? [0, %d]  ', maxLayers);
    raw_answer = input('');
    answer = int32(raw_answer);
    if length(answer) > 1
        layers = answer(1);
    else
        layers = answer;
    end
    %boundary checking for permitted layers
    if layers > maxLayers
        layers = maxLayers;
    elseif layers <= 0
        layers = 0;
    end
end

%exit menu if there are no hidden layers
if layers > 0
    % get the number of nodes in each layer
    for i = 1:layers
        nodeCount = 0;
        answer = -1;
        while(answer < 0)
            fprintf('(Layer %d) Enter number of nodes [0, %d]',...
                i, maxNodes);
            fprintf('\n**Enter 0 for the default**  >> ');
            raw_answer = input('');
            answer = int32(raw_answer);
            if length(answer) > 1
                nodeCount = answer(1);
            else
                nodeCount = answer;
            end
            
            % boundary check for nodes per layer
            if (nodeCount > maxNodes)
                nodeCount = maxNodes;
            elseif (nodeCount < 0)
                nodeCount = 1;
            end
            nodesPerLayer = [nodesPerLayer nodeCount];
        end
    end
else
    nodesPerLayer(1) = 0;
end


end