# engr240_nn
MATLAB based MLP
This is the project for my Engineering Applications class for Fall 2019.
Last update: 7 Nov 2019

NOTES:
- need to add error history to show changes to weights
- need to add graph function for aforementioned
- relu generates NaN costs during backprop
- persistent error in Network/trainNetwork:  Too many input arguments.


Functions:
- activate: activation functions
- backprop: backprop algorithm
- cost: cost functions
- createNetwork: initializes the neural net
- displayNetworkDesign: allows user to numerically see how the network is designed
- feedforward: simple feedforward process(weights * input + data)
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
