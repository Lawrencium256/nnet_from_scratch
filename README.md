# Building Neural Network from Scratch

Project to build a neural network from scratch in Python.

Basic libraries such as Numpy, Random and Matplotlib are used but deep learning packages were avoided during the design process. PyTorch is only used to verify model accuracy.

This vanilla feedforward model has been trained on MNIST data, using MSE loss and sigmoid activation. The next ways to add to the model include (but are not limited to):

* Choice of activation function (this has been added but not integrated into `Network()` class).
* Choice of loss.
* Different parameter initialisation
* Regularisation (e.g. L2).
* Early stopping.
* Sparsity penalty.
* Adaptive learning rate.

The model has been trained to learn MNIST with approx. 90% accuracy after 1000 with a 50k/10k training/test split. This performance is not as good as expected (from comparison with other works) but the underlying reason is yet to be determined.
