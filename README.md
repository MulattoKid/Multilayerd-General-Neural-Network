# Multilayerd-General-Neural-Network
This is my personal Multilayered General Neural Network project built from scratch. Currently, the only file needed is *NN.py*. The implementation utilizes Python with NumPy and Matplotlib (only needed if you want to plot the error) as dependencies. These can be installed with the following commands:
```
pip install numpy
pip install matplotlib
```
For great slides on how neural networks do their thing, check out Keith Downing's (my professor) [slides](http://www.idi.ntnu.no/emner/it3105/tdt76/) from the Deep Learning Theory Module held the autumn of 2017 at NTNU (especially lecture 2).

## Status
The current code base should support any basic neural network configuration. By basic, I mean the following:
* Multilayered networks
* 1 to *n* nodes per layer
* *n* fully-connected layers
* Static learning rate
* Runs for *n* epochs, regardless of error convergence

### Other features
* Perform validation testing every *n* epoch
* Dropout

### Activations functions
Currently, the following activation functions are supported:
* Logistic

### Loss functions
Currently, the following loss functions are supported:
* Squared error

## Usage
```python
nn = NN.NeuralNetwork(layer_sizes=[8, 32, 16, 4], learning_rate=0.1)
nn.Train(20, training_cases, validation_interval=100)
nn.Test(test_cases)
nn.PlotError()
```
The *case_base* must have the following structure:
```python
case_0 = 
[
  [[input_0, input_1, ...]], #Input
  [target_0, target_1, ...] #Expected output/target
]
case_1 = [...]
case_base = [case_0, case_1, ...]
```
No parameters are needed when running the script.

**Warning:** I've tested the implementation, and as far as I can tell it works as intented. However, unknown errors might be present.
