import sys
sys.path.append("..")
import NN

def OR():
    print("\nRunning OR Neural Network:")

    case_0 = [[[0, 0]], [0]]
    case_1 = [[[1, 0]], [1]]
    case_2 = [[[0, 1]], [1]]
    case_3 = [[[1, 1]], [1]]
    case_base = [case_0, case_1, case_2, case_3]

    nn = NN.NeuralNetwork(layer_sizes=[2, 1], learning_rate=0.1)
    nn.Train(1000, case_base, print_interval=100000)
    #print("Final weights:\n", nn.layers[nn.output_layer].weights)
    nn.Test(add_bias=False)
    nn.PlotError()

def AND():
    print("\nRunning AND Neural Network:")

    case_0 = [[[0, 0]], [0]]
    case_1 = [[[1, 0]], [0]]
    case_2 = [[[0, 1]], [0]]
    case_3 = [[[1, 1]], [1]]
    case_base = [case_0, case_1, case_2, case_3]

    nn = NN.NeuralNetwork(layer_sizes=[2, 1], learning_rate=0.1)
    nn.Train(1000, case_base, print_interval=100000)
    #print("Final weights:\n", nn.layers[nn.output_layer].weights)
    nn.Test(add_bias=False)
    nn.PlotError()

def XOR():
    print("\nRunning XOR Neural Network:")

    case_0 = [[[0, 0]], [0]]
    case_1 = [[[1, 0]], [1]]
    case_2 = [[[0, 1]], [1]]
    case_3 = [[[1, 1]], [0]]
    case_base = [case_0, case_1, case_2, case_3]

    nn = NN.NeuralNetwork(layer_sizes=[2, 2, 1], learning_rate=0.1)
    nn.Train(10000, case_base, print_interval=100000)
    nn.Test(add_bias=False)
    nn.PlotError()

OR()
AND()
XOR()