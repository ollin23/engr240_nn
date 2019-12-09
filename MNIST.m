function [labels, images] = MNIST(data)
%MNIST takes the provided data and creates 10 grayscale sample images
% from the MNIST dataset and returns the label and images in separate data 
% structures.
% 
% This function is custom made for NIST data converted into CSV format
% which contains image labels as the first column.
%
% MNIST(data)
% PARAMETERS
% data : a csv data set containing MNIST data
%
% RETURN VALUES
% labels : the single digit target label values
% images : 28x28 numeric images normalized and transformed into a 1x784
%          vector


% get the dimensions of the data
[rows, cols] = size(data);

% store the target labels from the dataset
labels = data(:,1);

% store the image data
images = data(:,2:cols);

% % images size
sz = sqrt(cols-1);
fprintf('\nImages size: %d x %d pixels.\n',sz,sz);
fprintf('Total number of images is %d.\n',length(labels));
fprintf('\nPress any button to continue.\n');

% get visual samples of the data
colormap gray;
clf;
number_of_samples = 10;
step = rows / number_of_samples;
for i = 1:step:rows
    imagesc(reshape(images(i,:),sz,sz)'),...
        title(['Target Label: ',num2str(labels(i)),...
        '   (Press any button to continue)'])
    pause();
end

%normalize image data
images = images / 255;

close all;
end
