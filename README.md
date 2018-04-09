# Multilayerd-General-Neural-Network
This is my personal Multilayered General Neural Network project built from scratch. Currently, the only file needed is *NN.py*. The implementation utilizes Python with NumPy and Matplotlib (only needed if you want to plot the error) as dependencies. These can be installed with the following commands:
* pip install numpy
* pip install matplotlib

## Status
The current code base should support any basic neural network configuration. By basic, I mean the following:
* Multilayered networks
* 1 to *n* nodes per layer
* *n* fully-connected layers
* Static learning rate
* Runs for *n* epochs, regardless of error convergence

### Activations functions
Currently, the following activation functions are supported:
* Logistic

### Loss functions
Currently, the following loss functions are supported:
* Squared error

## Usage
```python
nn = NeuralNetwork(layer_sizes=[8, 4, 2, 1], learning_rate=0.1)
nn.Train(10000, case_base)
nn.Classify(case_base)
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
