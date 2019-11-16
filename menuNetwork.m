function [net] = menuNetwork(images, labels)
%menuNetwork provides the interface to create a new MLP
%
%PARAMETERS
% images - the image dataset to analyze
%
%OUTPUT
% net - the basic neural net

    % get size of output vector
    [sampleCount, imgSize] = size(images);
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
    net.encodedLabels = oneHotEncoding(labels);
    net.images = images;
    net.labels = labels;

    %randomize images
    [r,~] = size(net.images);
    shuffledIndex = randperm(r);
    net.images(1:end,:) = net.images(shuffledIndex,:); 
end