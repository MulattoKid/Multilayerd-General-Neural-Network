import sys
sys.path.append("..")
import NN

def GenOneHotVector(index, length):
    one_hot = [0 for x in range(length)]
    one_hot[index] = 1
    return one_hot

def LoadDigits(file):
    cases = []
    with open(file) as f:
        for line in f:
            line = line.replace('\n', '').split(',')
            line = [float(x) for x in line]
            case = [[[x / 16.0 for x in line[:-1]]], GenOneHotVector(int(line[len(line) - 1]), 10)] #Normalize inputs
            cases.append(case)
    return cases

print("\nRunning Digit Recognition Neural Network:")
training_cases = LoadDigits("digits.tra")
test_cases = LoadDigits("digits.tes")
nn = NN.NeuralNetwork(layer_sizes=[8*8, 32, 16, 10], learning_rate=0.1, apply_dropout=True, dropout_rate=0.2)
nn.Train(20, training_cases, validation_interval=5, print_interval=1)
nn.Test(test_cases)
nn.PlotError()