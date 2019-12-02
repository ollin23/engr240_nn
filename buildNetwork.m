function [net] = buildNetwork(images, labels)
%buildNetwork provides the interface to create a new MLP
%
%PARAMETERS
% images - the image dataset to analyze
% labels - image labels
%
%OUTPUT
% net - the basic neural net

    % get size of output vector
    [~, imgSize] = size(images);
    nodes = imgSize;
    outputCount = length(unique(labels));

    %determine default count of nodes in layer and number of layers
    counter = 1;
    layers(counter) = nodes;
    while (2*outputCount <= nodes)
        nodes = round(sqrt(layers(counter))) + outputCount;
        counter = counter + 1;
        layers(counter) = nodes;
    end
    layers = [layers outputCount];

    % get input from user for number of layers and the number of nodes in each
    % layer
    fprintf('Do you want to use the default design? (Y/n)  ');
    answer = input('','s');
    answer = lower(answer);
    if length(answer) < 1
        answer = 'y';
    end
    if (strcmpi(answer,'n') || strcmpi(answer(1),'n'))
        nodesEachLayer = menu();
        layers = [imgSize nodesEachLayer outputCount];
    end

    % create network
    net = Network(layers);

    % display the network topology
    displayNetworkDesign(layers);
    disp('<< Press Enter to continue...>>');
    % pause();

    % encode the labels
    % add network metric parameters
    net.imageData = images;
    net.labels = labels;

    %randomize images
    [r,~] = size(net.imageData);
    shuffledIndex = randperm(r);
    net.imageData(1:end,:) = net.imageData(shuffledIndex,:); 
    net.labels = net.labels(shuffledIndex);
    net.encodedLabels = oneHotEncoding(net.labels);
end