# engr240_nn
MATLAB based MLP
This is the project for my Engineering Applications class for Fall 2019.
Last update: 11 Nov 2019

NOTES:
- massive update. Rewrote the backpropagation algorithm after encountering an intractable problem
  with the trajectory of the code base and the implementation of batches. This entailed the redesign
  of the feedforward, train, and update functions.
- the majority of code, blogs, and websites  tentatively referenced online have proven unfruitful. the
  overwhelming number of code was either for no hidden layer perceptrons, or hardcoded the relationships
  among the layers of the MLP. most of the remaining code was cryptically written with horrid documentation
  or just flat out wrong. on the theoretical side, i had to use the book *Deep Learning* (Goodfellow, Bengio,
  Courville; 2016) to get a meaningful and reproducible treatment of backprop for deep feedforward networks.
- substantial documentation still required, but most of the code is straightforward
- regularization has yet to be implemented
- still intend to fully implement OOP functions; currently receiving errors for all of them except the
  constructor function

Updates:
- backprop2: consideration of batches as a modification of SGD rather than treating them as the default
- feedforward2: linear transformation of data; aligned to object-oriented schema
- train2: trains neural net; realigned to the implementation of backprop2
- update2: aligned to object-oriented schema

Functions:
- activate: activation functions
- cost: cost functions
- createNetwork: initializes the neural net
- displayNetworkDesign: allows user to numerically see how the network is designed
- fit: wrapper function for training data; implements epoch cycles and graph
- menu: menu to drive topology design
- samples: splits the csv into label and image structures; shows user a sample of the images
- sigmoid: sigmoid function
- softmax: stabilized softmax function

Objects
- Network: the neural network object

Main Programs
- GetAndConvertFiles: downloads and converts binary MNIST files to csv into "projects" subfolder of the current directory
- ProjectMain: the main project file

ARCHIVE
Files
- backprop: backprop algorithm
- feedforward: simple feedforward process(weights * input + data)
- train: trains the neural net
- update: update the weights and biases

Notes, 8 Nov 2019:
- fixed relu errors; still potentially unstable
- fixed error history storage
- added graph of errors
