# Neural Network
# creates a new Neural network
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense, Activation, Flatten
from directkeys import PressKey, ReleaseKey, W, A, S, D, JUMP, RUN
import numpy as np
import math
import random
import time

class neural_network:
    def __init__(self, data, first, second, output, iter=1,
                weights1=np.zeros((4200, 64)),
                weights2=np.zeros((64, 16)),
                weights3=np.zeros((16, 6))):
        # self.iter = iter

        weights1_zeros = np.any(weights1)
        weights2_zeros = np.any(weights2)
        weights3_zeros = np.any(weights3)


        if (not weights1_zeros):
            weights1 = np.random.normal(size = (4200, 64))

        if (not weights2_zeros):
            weights2 = np.random.normal(size = (64, 16))

        if (not weights3_zeros):
            weights3 = np.random.normal(size = (16, 6))

        self.layer0 = {"biases": np.zeros(data)}
        self.layer1 = {"weights": weights1,
                       "biases": np.zeros(first)}
        self.layer2 = {"weights": weights2,
                       "biases": np.zeros(second)}
        self.layer3 = {"weights": weights3,
                       "biases": np.zeros(output)}
        # self.model = Sequential([
        #     #Flatten(),
        #     Dense(first, activation = "relu", input_dim = 1),
        #     Dense(second, activation = "relu"),
        #     Dense(output, activation = "softmax"),
        #     #Dense(output, activation = "softmax")
        #
        # ])

        self.fitness = 0

    def mutate(self, data, first, second, output):
        first_layer = np.zeros((data,first))
        second_layer = np.zeros((first, second))
        third_layer = np.zeros((second, output))
        i = 0
        j = 0
        for first_weight in self.layer1["weights"]:
            for next_weight in first_weight:
                if(random.random() < 0.25):
                    first_layer[i, j] = next_weight * random.uniform(-1, 1)
                else:
                    first_layer[i, j] = next_weight
                j += 1
            j = 0
            i += 1
        i = 0
        j = 0
        for second_weight in self.layer2["weights"]:
            for next_weight in second_weight:
                if(random.random() < 0.25):
                    second_layer[i, j] = next_weight * random.uniform(-1, 1)
                else:
                    second_layer[i, j] = next_weight
                j += 1
            j = 0
            i += 1
        i = 0
        j = 0
        for third_weight in self.layer3["weights"]:
            for next_weight in third_weight:
                if(random.random() < 0.25):
                    third_layer[i, j] = next_weight * random.uniform(-1, 1)
                else:
                    third_layer[i, j] = next_weight
                j +=  1
            j = 0
            i += 1
        mario = neural_network(data, first, second, output, 1, first_layer,
                               second_layer, third_layer)

        return mario

    def play(self, screen):
        test_screen = screen/255
        self.layer0["biases"] = test_screen
        np.matmul(self.layer0["biases"], self.layer1["weights"], self.layer1["biases"])
        self.layer1["biases"] = sigmoid(self.layer1["biases"])
        self.layer2["biases"] = np.matmul(self.layer1["biases"], self.layer2["weights"])
        self.layer2["biases"] = sigmoid(self.layer2["biases"])
        self.layer3["biases"] = np.matmul(self.layer2["biases"], self.layer3["weights"])
        self.layer3["biases"] = sigmoid(self.layer3["biases"])
        output = self.layer3["biases"]

        # output = np.array(self.model.predict([test_screen], 154425, verbose = 0)[0])
        #print(output.shape)
        #print(output)
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


def sigmoid(x):
    for j in range(len(x)):
        if (x[j] > 170):
            x[j] = 1
        elif (x[j] < -170):
            x[j] = 0
        else:
            x[j] = 1/(1+math.exp(-x[j]))
    return x

def forward():
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(W)
    ReleaseKey(JUMP)

def backward():
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    ReleaseKey(W)
    ReleaseKey(JUMP)

def jump():
    PressKey(JUMP)
    ReleaseKey(D)
    ReleaseKey(S)
    ReleaseKey(W)
    ReleaseKey(A)

def jump_forward():
        PressKey(D)
        PressKey(JUMP)
        ReleaseKey(D)
        ReleaseKey(S)
        ReleaseKey(W)

def jump_backward():
        PressKey(A)
        PressKey(JUMP)
        ReleaseKey(D)
        ReleaseKey(S)
        ReleaseKey(W)

def stay():
        ReleaseKey(A)
        ReleaseKey(JUMP)
        ReleaseKey(D)
        ReleaseKey(S)
        ReleaseKey(W)
