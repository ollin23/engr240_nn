# engr240_nn
MATLAB based MLP
This is the project for my Engineering Applications class for Fall 2019.
As of 4 Nov 2019ev, consists of 12 files. 9 functions with 2 main programs and 1 object file.

Functions:
- activate: activation functions
- cost: cost functions
- createNetwork: initializes the neural net
- displayNetworkDesign: allows user to numerically see how the network is designed
- feedforward: simple feedforward process(weights * input + data)
- menu: menu to drive the nn topology design
- samples: splits the csv into label and image structures; shows user a sample of the images
- sigmoid: sigmoid function
- softmax: stabilized softmax function

Object
- Network: the neural network object

Main Programs
- GetAndConvertFiles: downloads and converts binary MNIST files to csv into "projects" subfolder of the current directory
- ProjectMain: runs the neural net
