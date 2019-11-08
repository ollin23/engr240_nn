# engr240_nn
MATLAB based MLP
This is the project for my Engineering Applications class for Fall 2019.
Last update: 8 Nov 2019

NOTES:
- fixed relu errors; still potentially unstable
- fixed error history storage
- added graph of errors

Functions:
- activate: activation functions
- backprop: backprop algorithm
- cost: cost functions
- createNetwork: initializes the neural net
- displayNetworkDesign: allows user to numerically see how the network is designed
- feedforward: simple feedforward process(weights * input + data)
- fit: wrapper function for training data; implements epoch cycles and graph
- menu: menu to drive topology design
- samples: splits the csv into label and image structures; shows user a sample of the images
- sigmoid: sigmoid function
- softmax: stabilized softmax function
- train: trains the neural net

Object
- Network: the neural network object

Main Programs
- GetAndConvertFiles: downloads and converts binary MNIST files to csv into "projects" subfolder of the current directory
- ProjectMain: the main project file
