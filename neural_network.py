from directkeys import PressKey, ReleaseKey, W, A, S, D, L, JUMP, RUN
import numpy as np
import math
import random
import time
import cv2
import pyautogui as pag

# match_value is the return value from the opencv template match
# this minimum shows the minimum allowable return value when comparing the
# current screen to the previous screen. If the return value is less than this
# then mario is either dead or did not move this iteration
MATCH_VALUE_MIN = 0.005
NUM_TIMES_MIN_CAN_BE_EXCEDED = 3
MUTATION_CHANCE = 0.1
MAX_INDICES_TO_ADD = 5

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
        self.input_data_indices = set()
        self.input_layer_size = input_layer_size
        self.hidden_layers = hidden_layers
        self.alive = True
        self.num_times_min_exceded = 0
        self.fitness = 0
        self.screen_size = 10

    @staticmethod
    def compare_screens(screen, template, method=cv2.TM_SQDIFF_NORMED):
        """
        Compares 2 screens using openCV template match with cv2.TM_SQDIFF method.

        Arguments:
        screen - A numpy array representing the first screen to compate to template
        template - A numpy array representing the template that is being compared to the screen
        """
        comparison = cv2.matchTemplate(screen, template, method)
        # if the two screens are sufficiently similar mario either hasn't moved
        # or has died. So we increment a counter, if this happens many times
        # he is either walking into a wall or has died
        if comparison < MATCH_VALUE_MIN:
            self.num_times_min_exceded += 1
        return comparison


    def update(self, prev_screen, curr_screen):
        """
        Checks if the current mario is alive and updates his fitness value.

        Arguments:
        prev_screen: A numpy array that represents the screen from the previous iteration
        curr_screen: A numpy array that represents the screen from the current iteration
        """

        black_screen = np.zeros(curr_screen.shape, np.uint8)
        fitness_comparison = cv2.matchTemplate(curr_screen, prev_screen, cv2.TM_SQDIFF_NORMED)
        black_screen_comparison = cv2.matchTemplate(curr_screen, black_screen, cv2.TM_SQDIFF_NORMED)
        if fitness_comparison < MATCH_VALUE_MIN:
            # if mario has been dead or has not moved in this many iterations
            # we will say he has died
            self.num_times_min_exceded += 1
            print(f"\t\tMARIO HAS EARNED A STRIKE {NUM_TIMES_MIN_CAN_BE_EXCEDED - self.num_times_min_exceded} REMAINING BEFORE HE DIES")
        elif black_screen_comparison < MATCH_VALUE_MIN:
            self.num_time_min_exceded += 2
        else:
            fitness_comparison = cv2.matchTemplate(curr_screen, prev_screen, cv2.TM_SQDIFF_NORMED)
            self.fitness += fitness_comparison
        if self.num_times_min_exceded >= NUM_TIMES_MIN_CAN_BE_EXCEDED:
            self.alive = False

    def mutate_input_data(self):
        """
        Randomly adds between 1 and 5 indices to the input_data_indices list
        """
        num_to_add = random.randint(1, MAX_INDICES_TO_ADD)
        indices = []
        added = 0
        while added < num_to_add:
            num = random.randint(0, self.screen_size)
            if num not in self.input_data_indices:
                self.input_data_indices.add(num)
                added += 1
        self.input_layer_size += num_to_add
        self.brain["input_layer"]["biases"] = np.zeros(self.input_layer_size)
        new_weights = None
        layer = "output_layer"
        if "hidden_layer_0" in self.brain:
            layer = "hidden_layer_0"
        new_weights = np.random.normal(size = (num_to_add, self.brain[layer]["weights"].shape[1]))
        self.brain[layer]["weights"] = np.append(self.brain[layer]["weights"], new_weights, axis=0)

    def get_input_layer_from_screen(self, screen):
        while len(self.input_data_indices) != self.input_layer_size:
            # we need to determine what our input will be the input should be
            # a random set of pixels. This will also have a chance to mutate
            num = random.randint(0, len(screen))
            if num not in self.input_data_indices:
                self.input_data_indices.add(num)
        draw_screen = np.reshape(screen, (1080, 1920), order='C')
        # for point in self.input_data_indices:
        #     x = point % 1920
        #     y = point // 1920
        #     draw_screen = cv2.circle(draw_screen, (x,y), radius=0, color=(0, 0, 255), thickness=10)
        #     cv2.imshow('frame',draw_screen)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break
        return [screen[i] for i in self.input_data_indices]

    def play(self, screen):
        """
        Multiplies all the weights and biases at each layer and produces the output

        Arguments:
        screen: A flattened numpy array representing the screen the game is being played on
        """

        input_data = self.get_input_layer_from_screen(screen)
        self.brain["input_layer"]["biases"] = self.sigmoid(input_data)
        prev_biases = self.brain["input_layer"]["biases"]

        for hidden_layer in range(len(self.hidden_layers)):
            curr_layer = self.brain[f"hidden_layer_{hidden_layer}"]
            curr_layer["biases"] = np.matmul(prev_biases, curr_layer["weights"])
            curr_layer["biases"] = self.sigmoid(curr_layer["biases"])
            prev_biases = curr_layer["biases"]
        curr_layer = self.brain["output_layer"]
        curr_layer["biases"] = np.matmul(prev_biases, curr_layer["weights"])
        output = self.sigmoid(curr_layer["biases"])

        if(output[3] > 0.5 and output[4] <= 0.5):
            self.forward()
        elif(output[1] > 0.5 and output[4] <= 0.5):
            self.backward()
        elif(output[3] > 0.5 and output[4] > 0.5):
            self.jump_forward()
        elif(output[1] > 0.5 and output[4] > 0.5):
            self.jump_backward()
        elif(output[4] > 0.5):
            self.jump()
        else:
            self.stay()

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
        # PressKey(D)
        # ReleaseKey(A)
        # ReleaseKey(S)
        # ReleaseKey(W)
        # ReleaseKey(JUMP)
        pag.keyDown('d')
        pag.keyUp('a')
        pag.keyUp('s')
        pag.keyUp('w')
        pag.keyUp('space')

    @staticmethod
    def backward():
        pag.keyDown('a')
        pag.keyUp('d')
        pag.keyUp('s')
        pag.keyUp('w')
        pag.keyUp('space')
        # PressKey(A)
        # ReleaseKey(D)
        # ReleaseKey(S)
        # ReleaseKey(W)
        # ReleaseKey(JUMP)

    @staticmethod
    def jump():
        pag.keyDown('space')
        pag.keyUp('d')
        pag.keyUp('s')
        pag.keyUp('w')
        pag.keyUp('a')
        # PressKey(JUMP)
        # ReleaseKey(D)
        # ReleaseKey(S)
        # ReleaseKey(W)
        # ReleaseKey(A)

    @staticmethod
    def jump_forward():
        pag.keyDown('d')
        pag.keyDown('space')
        pag.keyUp('s')
        pag.keyUp('w')
        pag.keyUp('a')


    @staticmethod
    def jump_backward():
        pag.keyDown('a')
        pag.keyDown('space')
        pag.keyUp('s')
        pag.keyUp('w')
        pag.keyUp('d')
        # PressKey(A)
        # PressKey(JUMP)
        # ReleaseKey(D)
        # ReleaseKey(S)
        # ReleaseKey(W)

    @staticmethod
    def stay():
        pag.keyUp('d')
        pag.keyUp('space')
        pag.keyUp('s')
        pag.keyUp('w')
        pag.keyUp('a')
        # ReleaseKey(A)
        # ReleaseKey(JUMP)
        # ReleaseKey(D)
        # ReleaseKey(S)
        # ReleaseKey(W)
