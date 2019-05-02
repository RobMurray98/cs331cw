# cs331cw
Source code for three-layer MLP for the XOR problem.

## ann.py

Provides a class for an three-layer MLP for the XOR problem.

When executed as a script using `python ann.py`, trains an MLP using fixed noisy training vectors, plotting the output of the trained network over a regular array of input vectors covering the unit square and mean squared error against epoch number.

## main.py

When executed as a script using `python ann.py`, trains an MLP with 2, 4 and 8 hidden layer nodes and 16, 32 and 64 noisy vectors in the training data, displaying and saving plots for each of the 9 cases.
