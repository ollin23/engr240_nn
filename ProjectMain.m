% This is the main script file for the project
clear; clc;
% clear layers nodes outputCount response counter;
format compact;
% % actual data
% data = load('train_data.csv');
datadir = [pwd '\project\test_samples.csv'];
data = load(datadir);
 
% split data file into images and labels
[labels, images] = samples(data);

% get size of output vector
[sampleCount, imgSize] = size(images);
nodes = imgSize;
targets = unique(labels);
outputCount = length(targets);


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
if (strcmp(answer,'n') || strcmp(answer(1),'n'))
    nodesEachLayer = menu();
    layers = [imgSize nodesEachLayer outputCount];
end

% create network 
nn = Network(layers);

% display the network topology
displayNetworkDesign(layers);
disp('<< Press Enter to continue...>>');
pause();

for i = 1:sampleCount
    
end
