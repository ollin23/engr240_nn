# engr240_nn
MATLAB based MLP
This is the project for my Engineering Applications class for Fall 2019.
Last update: 15 Nov 2019

NOTES:
- Network: updated reporting, added split function
- firstMenu: menu on start of program to choose dataset
- improved R2 calculation. previous implementation increased the runtime 10x
- other functions updated for compatibility


Functions:
- activate: activation functions
- backprop2: backpropagation algorithm
- cost: cost functions
- createNetwork: initializes the neural net
- displayNetworkDesign: allows user to numerically see how the network is designed
- feedforward2: feedforward algorithm
- firstMenu: first options for using project, choice of which dataset to use
- fit2: wrapper function for training data; implements epoch cycles and graph
- menu: menu to drive topology design
- menuHyper: menu to drive hyperpamater tuning
- oneHotEncoding: encodes input data
- MNIST: splits the csv into label and image structures; shows user a sample of the images
- sigmoid: sigmoid function
- softmax: softmax function
- split: splits data into training, validation, and testing sets
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
- fit: replaced by fit2

Notes, 15 Nov 2019
- fit2: updated with early stop, longmemory, r2, and more plots implementation
- Network:
    * properties added -
        o trial - keeps track of trial runs (one trial = 1 set of training through all epochs)
        o r2 - R-squared, stored per trial, tracked per epoch
        o longmemory - history of output throughout a trial
    * functions added -
        o report() and timedReport() - generate text reports and save the network in a mat file
        o reset() - resets the network to original parameters
        o fit() - updated to include dropout option
- train2: updated with longmemory property, optimizations tested
- cost: updated with derivative parameter
- pred: prediction function
- normalize: function to normalize data
- activate: minor documentation update


Notes, 12 Nov 2019
- Changed optimization implementation. A struct was more useful than several independent variables.
- Still in process of implementing two more gradient optimizations.
- A menu to select optimization features will be added as well as options to save the
  visualizations generated at the end of training.
- Designing prediction function to work non-training data.

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

Notes, 8 Nov 2019:
- fixed relu errors; still potentially unstable
- fixed error history storage
- added graph of errors
