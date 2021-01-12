from directkeys import PressKey, ReleaseKey, W, A, S, D, JUMP, RUN
import numpy as np
import math
import random
import time


MUTATION_CHANCE = 0.1

class NeuralNetwork:
    """
    Creates a neural network. The user can specify the number of layers they
    want as well as the size of each layer.
    """
    def __init__(self, input_layer_size, output_layer_size, hidden_layers):
        """
        creates a neural network with the given input and output layer sizes.
        user can also optionally add as many hidden layers as they want.

        Arguments:
        input_layer_size: an integer representing the size of the input
        output_layer_size: an integer representing the size of the output
        hidden_layers: a list of integers that represent the size of a hidden layer.
                       The first element will be the size of the first hidden layer
                       and the next element will be the next layer and so on.
        """
        neural_net = {}
        neural_net["input_layer"] = {"biases": np.zeros(input_layer_size)}
        prev_layer_size = input_layer_size
        for i, hidden_layer_size in enumerate(hidden_layers):
            weights = np.random.normal(size = (prev_layer_size, hidden_layer_size))
            neural_net[f"hidden_layer_{i}"] = {"weights": weights,
                                               "biases": np.zeros(hidden_layer_size)}
            prev_layer_size = hidden_layer_size
        weights = np.random.normal(size = (prev_layer_size, output_layer_size))
        neural_net[f"output_layer"] = {"weights": weights,
                                       "biases": np.zeros(output_layer_size)}
        self.brain = neural_net
        self.fitness = -1


    def mutate(self):
        """
        Changes the weights of the NeuralNetwork. The chance of mutation
        is defined by the MUTATION_CHANCE global constant. Approximately
        that percentage of weights will be changed. The weights are multiplied
        by a random number between 1 and -1 and that is the new weight.
        """
        for layer in self.brain:
            try:
                for weight_list in self.brain[layer]["weights"]:
                    for i, weight in enumerate(weight_list):
                        if random.random() < MUTATION_CHANCE:
                            weight = weight * random.uniform(-1, 1)
                            weight_list[i] = weight
            except KeyError:
                # the input layer doesn't have a weights field so we just pass
                pass



class Mario(NeuralNetwork):
    """
    A class that uses a NeuralNetwork to play mario
    """
    def __init__(self, input_layer_size, output_layer_size, hidden_layers):
        super().__init__(input_layer_size, output_layer_size, hidden_layers)
        self.input_data = []
        self.input_layer_size = input_layer_size
        self.hidden_layers = hidden_layers

    def play(self, screen):
        """
        Multiplies all the weights and biases at each layer and produces the output

        Arguments:
        screen: A flattened numpy array representing the screen the game is being played on
        """
        if not self.input_data:
            # we need to determine what our input will be the input should be
            # a random set of pixels. This will also have a chance to mutate
            indices = np.random.choice(screen, self.input_layer_size)
            for index in indices:
                self.input_data.append(screen[index])
        self.brain["input_layer"]["biases"] = sigmoid(self.input_data)
        prev_biases = self.brain["input_layer"]["biases"]

        for hidden_layer in range(len(self.hidden_layers)):
            curr_layer = self.brain[f"hidden_layer_{hidden_layer}"]
            curr_layer["biases"] = np.matmul(prev_biases, curr_layer["weights"])
            curr_layer["biases"] = sigmoid(curr_layer["biases"])
            prev_biases = curr_layer["biases"]
        curr_layer = self.brain["output_layer"]
        curr_layer["biases"] = np.matmul(prev_biases, curr_layer["weights"])
        output = sigmoid(curr_layer["biases"])
        print(output)

        if(output[3] > 0.5 and output[4] <= 0.5):
            forward()
        elif(output[1] > 0.5 and output[4] <= 0.5):
            backward()
        elif(output[3] > 0.5 and output[4] > 0.5):
            jump_forward()
        elif(output[1] > 0.5 and output[4] > 0.5):
            jump_backward()
        elif(output[4] > 0.5):
            jump()
        else:
            stay()

    @staticmethod
    def sigmoid(x):
        """
        Function that maps the input value, x, to a value between 1 and 0
        """
        for j in range(len(x)):
            if (x[j] > 170):
                x[j] = 1
            elif (x[j] < -170):
                x[j] = 0
            else:
                x[j] = 1/(1+math.exp(-x[j]))
        return x

    @staticmethod
    def forward():
        PressKey(D)
        ReleaseKey(A)
        ReleaseKey(S)
        ReleaseKey(W)
        ReleaseKey(JUMP)

    @staticmethod
    def backward():
        PressKey(A)
        ReleaseKey(D)
        ReleaseKey(S)
        ReleaseKey(W)
        ReleaseKey(JUMP)

    @staticmethod
    def jump():
        PressKey(JUMP)
        ReleaseKey(D)
        ReleaseKey(S)
        ReleaseKey(W)
        ReleaseKey(A)

    @staticmethod
    def jump_forward():
        PressKey(D)
        PressKey(JUMP)
        ReleaseKey(D)
        ReleaseKey(S)
        ReleaseKey(W)

    @staticmethod
    def jump_backward():
        PressKey(A)
        PressKey(JUMP)
        ReleaseKey(D)
        ReleaseKey(S)
        ReleaseKey(W)

    @staticmethod
    def stay():
        ReleaseKey(A)
        ReleaseKey(JUMP)
        ReleaseKey(D)
        ReleaseKey(S)
        ReleaseKey(W)
