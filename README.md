# engr240_nn
MATLAB based MLP
This is the project for my Engineering Applications class for Fall 2019.
Last update: 12 Nov 2019

NOTES
- Changed optimization implementation. A struct was more useful than several independent variables.
- Still in process of implementing two more gradient optimizations.
- A menu to select optimization features will be added as well as options to save the
  visualizations generated at the end of training.
- Designing prediction function to work non-training data.


Updates:
- backprop2: added some optimization options
- train2: added some optimization options; tweaked the accuracy and error metrics
- MNIST: renamed from samples

Functions:
- activate: activation functions
- backprop2: backpropagation algorithm
- cost: cost functions
- createNetwork: initializes the neural net
- displayNetworkDesign: allows user to numerically see how the network is designed
- feedforward2: feedforward algorithm
- fit: wrapper function for training data; implements epoch cycles and graph
- menu: menu to drive topology design
- menuHyper: menu to drive hyperpamater tuning
- oneHotEncoding: encodes input data
- MNIST: splits the csv into label and image structures; shows user a sample of the images
- sigmoid: sigmoid function
- softmax: softmax function
- train2: trains the network, gathers loss and accuracy metrics
- update2: updates network parameters

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

Notes, 11 Nov 2019:
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
