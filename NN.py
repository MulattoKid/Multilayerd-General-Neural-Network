import numpy as np
from matplotlib import pyplot as plt

class Layer:
    def __init__(self, layer_num, num_inputs, num_nodes, is_output_layer):
        self.layer_num = layer_num
        self.inputs = np.zeros(shape=(1, num_inputs + 1)) #+1 for bias term
        self.weights = np.zeros(shape=(num_inputs + 1, num_nodes)) #+1 for bias term
        self.deltas = np.zeros(shape=(num_nodes)) #Stores
        self.outputs = np.zeros(shape=(1, num_nodes))
        self.is_output_layer = is_output_layer

        #Weight initialization
        weight_range = 1.0 / np.sqrt(num_inputs)
        for y in range(self.weights.shape[0]):
            for x in range(self.weights.shape[1]):
                self.weights[y][x] = np.random.uniform(-weight_range, weight_range)

    def Print(self):
        print()
        print("Layer number:", self.layer_num)
        print("Input shape:", self.inputs.shape)
        print("Weights shape:", self.weights.shape)
        print("Deltas:", self.deltas.shape[0])
        print("Output shape:", self.outputs.shape)
        print("Is output layer:", self.is_output_layer)

    def UpdateOutput(self, inputs):
        self.inputs = inputs
        self.outputs = np.matmul(self.inputs, self.weights)
        self.outputs = 1.0 / (1.0 + np.exp(-self.outputs)) #Logistic activation function
        self.outputs = np.append(self.outputs, [[-1]], axis=1) #Add bias term to output so that the next layer has it as part of its input

class NeuralNetwork:
    def __init__(self, layer_sizes, learning_rate):
        self.num_layers = len(layer_sizes) - 1 #Disregard input layer
        self.output_layer = self.num_layers - 1
        self.num_hidden_layers = self.num_layers - 1
        self.num_inputs = layer_sizes[0] #Size of input layer
        self.layer_sizes = layer_sizes[1:] #Specifies the number of nodes in each hidden layer
        self.CreateLayers()
        self.learning_rate = learning_rate
        self.errors = []
        self.Print()

    def Print(self):
        print("Number of layers:", self.num_layers, "(+1 if you include input layer)")
        print("Number of hidden layers:", self.num_hidden_layers)
        print("Number of inputs:", self.num_inputs)
        print("Sizes of layers:", self.layer_sizes)
        for layer in self.layers:
            layer.Print()
        print()

    def CreateLayers(self):
        self.layers = []
        self.layers.append(Layer(0, self.num_inputs, self.layer_sizes[0], self.num_layers == 1)) #Take special care of 1st layer as it depends on the size of the input layer
        for i in range(1, self.num_layers): #Create remaining layers
            self.layers.append(Layer(i, self.layer_sizes[i - 1], self.layer_sizes[i], i == self.num_layers - 1))

    def FeedForward(self, inputs):
        self.layers[0].UpdateOutput(inputs) #Perform manual update on first layer
        for i in range(1, self.num_layers): #Update remaining layers
            self.layers[i].UpdateOutput(self.layers[i - 1].outputs)

    def Backpropagation(self, targets):
        outputs = np.reshape(self.layers[self.output_layer].outputs, newshape=(self.layers[self.output_layer].outputs.shape[1])) #Remove second dimension of output
        outputs = outputs[:-1] #Remember we add a bias "node" to each layer's output -> remove bias from output so that it is not taken into account when calculating error
        error = np.sum(0.5 * np.power((targets - outputs), 2)) #Squared Error
        self.errors.append(error)

        #Take special care of the output layer's weights
        for node in range(self.layers[self.output_layer].weights.shape[1]): #For each node in output layer -> column in weight matrix
            for weight in range(self.layers[self.output_layer].weights.shape[0]): #For each weight connected to this node -> row in weight matrix
                input_v = self.layers[self.output_layer].inputs[0][weight] #Get input to current weight
                target_v = targets[node]
                output_v = outputs[node]
                delta = (target_v - output_v) * output_v * (1.0 - output_v) * input_v
                self.layers[self.output_layer].weights[weight][node] += self.learning_rate * delta

    def BackpropagationMultiLayer(self, targets):
        outputs = np.reshape(self.layers[self.output_layer].outputs, newshape=(self.layers[self.output_layer].outputs.shape[1])) #Remove second dimension of output
        outputs = outputs[:-1] #Remember we add a bias "node" to each layer's output -> remove bias from output so that it is not taken into account when calculating error
        error = np.sum(0.5 * np.power((targets - outputs), 2)) #Squared Error
        #self.errors.append(error)

        #Take special care of the output layer's weights
        for node in range(self.layers[self.output_layer].weights.shape[1]): #For each node in output layer -> column in weight matrix
            target_v = targets[node]
            output_v = outputs[node]
            #Calculate delta-value for current node
            self.layers[self.output_layer].deltas[node] = output_v * (1.0 - output_v) * (target_v - output_v)
            for weight in range(self.layers[self.output_layer].weights.shape[0]): #For each weight connected to this node -> row in weight matrix
                input_v = self.layers[self.output_layer].inputs[0][weight] #Get input to current weight
                #Calculate update for each weight for given node
                weight_delta = self.layers[self.output_layer].deltas[node] * input_v
                self.layers[self.output_layer].weights[weight][node] += self.learning_rate * weight_delta

        for layer in range(self.output_layer - 1, -1, -1): #Iterate over remaining layers, starting from the second to last to the first (excluding input layer)
            for node in range(self.layers[layer].weights.shape[1]): #For each node in layer -> column in weight matrix
                #Calculate delta-value for current node
                node_output = self.layers[layer].outputs[0][node]
                self.layers[layer].deltas[node] = node_output * (1.0 - node_output)
                next_layer = layer + 1
                delta_addition = 0.0
                for weight in range(self.layers[next_layer].weights.shape[1]): #For each weight that connect the current node to a node in next_layer -> iterates through a row in the weight matrix of next_layer
                    delta_addition += self.layers[next_layer].weights[node][weight] * self.layers[next_layer].deltas[weight]
                self.layers[layer].deltas[node] *= delta_addition

                for weight in range(self.layers[layer].weights.shape[0]): #For each weight that connect the current node to a node in layer -> row in weight matrix
                    input_v = self.layers[layer].inputs[0][weight] #Get input to current weight
                    #Calculate update for each weight for given node
                    weight_delta = self.layers[layer].deltas[node] * input_v
                    self.layers[layer].weights[weight][node] += self.learning_rate * weight_delta

        return error

    def Train(self, epochs, case_base):
        #Add bias of -1 to each case
        for c in range(len(case_base)):
            case_base[c][0][0].append(-1.0)

        for i in range(epochs):
            epoch_error = 0.0
            for case in case_base:
                self.FeedForward(case[0])
                epoch_error += self.BackpropagationMultiLayer(case[1])
            self.errors.append(epoch_error / len(case_base))

    def Classify(self, cases):
        for case in cases:
            self.FeedForward(case[0])
            print()
            print("Case:")
            print("Input:", case[0][0][:-1]) #Remove bias from print
            print("Target:", case[1])
            print("Output: ", np.round(self.layers[self.output_layer].outputs[0][:-1]), " \n\t(raw: ", self.layers[self.output_layer].outputs[0][:-1], ")", sep='') #Remember to remov ebias term from output for ease of readability

    def PlotError(self):
        x = np.arange(0, len(self.errors))
        y = self.errors
        plt.xlabel('Epoch')
        plt.ylabel('Squared Error')
        plt.plot(x, y)
        plt.show()

def OR():
    print("\nRunning OR Neural Network:")

    case_0 = [[[0, 0]], [0]]
    case_1 = [[[1, 0]], [1]]
    case_2 = [[[0, 1]], [1]]
    case_3 = [[[1, 1]], [1]]
    case_base = [case_0, case_1, case_2, case_3]

    nn = NeuralNetwork(layer_sizes=[2, 1], learning_rate=0.1)
    nn.Train(1000, case_base)
    #print("Final weights:\n", nn.layers[nn.output_layer].weights)
    nn.Classify(case_base)

def AND():
    print("\nRunning AND Neural Network:")

    case_0 = [[[0, 0]], [0]]
    case_1 = [[[1, 0]], [0]]
    case_2 = [[[0, 1]], [0]]
    case_3 = [[[1, 1]], [1]]
    case_base = [case_0, case_1, case_2, case_3]

    nn = NeuralNetwork(layer_sizes=[2, 1], learning_rate=0.1)
    nn.Train(1000, case_base)
    #print("Final weights:\n", nn.layers[nn.output_layer].weights)
    nn.Classify(case_base)

def XOR():
    print("\nRunning XOR Neural Network:")

    case_0 = [[[0, 0]], [0]]
    case_1 = [[[1, 0]], [1]]
    case_2 = [[[0, 1]], [1]]
    case_3 = [[[1, 1]], [0]]
    case_base = [case_0, case_1, case_2, case_3]

    nn = NeuralNetwork(layer_sizes=[2, 2, 1], learning_rate=0.1)
    nn.Train(10000, case_base)
    nn.Classify(case_base)
    nn.PlotError()

#OR()
#AND()
#XOR()