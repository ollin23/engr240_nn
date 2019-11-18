function [net] = menuNetwork(images, labels)
%menuNetwork provides the interface to create a new MLP
%
%PARAMETERS
% images - the image dataset to analyze
% labels - image labels
% GPU - boolean; enables or disables GPU utilization when applicable
%
%OUTPUT
% net - the basic neural net

    try
        GPU = true;
        arbitrary = gpuArray(1);
    catch
        disp('This system cannot use GPU acceleration.');
        disp('Performance will be negatively affected.');
        GPU = false;
    end
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
    net = Network(layers, GPU);

    % display the network topology
    displayNetworkDesign(layers);
    disp('<< Press Enter to continue...>>');
    % pause();

    % encode the labels
    % add network metric parameters
    if net.GPU
        enc = oneHotEncoding(labels);
        net.encodedLabels = gpuArray(enc);
        net.images = gpuArray(images);
        net.labels = gpuArray(labels);
    else
        net.encodedLabels = oneHotEncoding(labels);
        net.images = images;
        net.labels = labels;
    end

    %randomize images
    [r,~] = size(net.images);
    shuffledIndex = randperm(r);
    net.images(1:end,:) = net.images(shuffledIndex,:); 
end