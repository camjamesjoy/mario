import numpy as np
import math
import random
import time
import cv2
from pynput.keyboard import Controller
from keys import Keys

# match_value is the return value from the opencv template match
# this minimum shows the minimum allowable return value when comparing the
# current screen to the previous screen. If the return value is less than this
# then mario is either dead or did not move this iteration
MIN_PIXELS_MOVED = 50
NUM_TIMES_MIN_CAN_BE_EXCEDED = 5
MUTATION_CHANCE = 0.1
MAX_INDICES_TO_ADD = 5
FORWARD = 0
BACKWARD = 1
JUMP = 2
HEIGHT = 3
SLICE_WIDTH = 5
MAX_PIXELS_POSSIBLE = 700
UPDATE_TIMER = 0.5

keyboard = Controller()

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
    def __init__(self, input_layer_size, output_layer_size, hidden_layers, screen_size):
        super().__init__(input_layer_size, output_layer_size, hidden_layers)
        self.input_data_indices = set()
        self.input_layer_size = input_layer_size
        self.hidden_layers = hidden_layers
        self.alive = True
        self.num_times_min_exceded = 0
        self.fitness = 0
        self.screen_size = screen_size
        self.prev_slice = np.array([])

    def update(self, x_start, y_start, x_end, y_end, conn):
        """
        Checks if the current mario is alive and updates his fitness value.
        Fitness value is an estimate of how many pixels the screen has shifted

        Arguments:
        curr_screen: A numpy array that represents the gray screen from the current iteration
        """
        from PIL import ImageGrab as im
        while self.alive:
            start = time.time()
            curr_screen = im.grab(bbox=(x_start, y_start, x_end, y_end))
            curr_screen = cv2.cvtColor(np.array(curr_screen), cv2.COLOR_RGB2GRAY)
            min_non_zeros = curr_screen.shape[0] * SLICE_WIDTH
            min_col = -1
            if self.prev_slice.size != 0:
                for i in range(curr_screen.shape[1] - SLICE_WIDTH):
                    curr_slice = curr_screen[:,i:i+SLICE_WIDTH]
                    diff = curr_slice - self.prev_slice
                    non_zeros = np.count_nonzero(diff)
                    pixels_moved = curr_screen.shape[1] - SLICE_WIDTH - i
                    if non_zeros <= min_non_zeros and pixels_moved < MAX_PIXELS_POSSIBLE:
                        min_non_zeros = non_zeros
                        min_col = i
            self.prev_slice = curr_screen[:,-SLICE_WIDTH:]
            if min_col == -1:
                continue
            pixels_moved = curr_screen.shape[1] - SLICE_WIDTH - (min_col + 1)
            self.fitness += pixels_moved * UPDATE_TIMER
            if pixels_moved < MIN_PIXELS_MOVED:
                self.num_times_min_exceded += 1
                print(f"\t\tMARIO HAS EARNED A STRIKE BECAUSE HE ONLY TRAVELED {pixels_moved} PIXELS, {NUM_TIMES_MIN_CAN_BE_EXCEDED - self.num_times_min_exceded} REMAINING UNTIL HE DIES")
            else:
                self.num_times_min_exceded = 0 # reset the min if mario moved forward more then the min number of pixels
                print("\t\t RESETING COUNTER BECAUSE MARIO MOVED FORWARD")
            if self.num_times_min_exceded >= NUM_TIMES_MIN_CAN_BE_EXCEDED:
                self.alive = False
                conn.send([False, self.fitness])
                conn.close()
            end = time.time()
            while end - start < UPDATE_TIMER:
                end = time.time()
            start = end

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
        return [screen[i] for i in self.input_data_indices]

    def draw_screen(self, screen):
        """
        takes the screen and draws the current inputs to the NN.
        """

        #draw_screen = np.reshape(screen, (1080, 1920), order='C')
        while True: #for point in self.input_data_indices:
            #x = point % 1920
            #y = point // 1920
            #draw_screen = cv2.circle(draw_screen, (x,y), radius=0, color=(0, 0, 255), thickness=10)
            cv2.imshow('frame',screen)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def play(self, screen):
        """
        Multiplies all the weights and biases at each layer and produces the output

        Arguments:
        screen: A flattened numpy array representing the screen the game is being played on
        """
        self.screen_size = len(screen)
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

        next_action = self.stay
        if(output[FORWARD] >= 0.5 and output[BACKWARD] < 0.5 and output[JUMP] < 0.5):
            next_action = self.forward
        elif(output[FORWARD] < 0.5 and output[BACKWARD] >= 0.5 and output[JUMP] < 0.5):
            next_action = self.backward
        elif(output[FORWARD] >= 0.5 and output[BACKWARD] < 0.5 and output[JUMP] >= 0.5):
            next_action = self.jump_forward
        elif(output[FORWARD] < 0.5 and output[BACKWARD] >= 0.5 and output[JUMP] >= 0.5):
            next_action = self.jump_backward
        elif(output[FORWARD] < 0.5 and output[BACKWARD] < 0.5 and output[JUMP] >= 0.5):
            next_action = self.jump


        if next_action == self.jump or next_action == self.jump_backward or next_action == self.jump_forward:
            next_action(output[HEIGHT] / 2)
        else:
            next_action()

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
        keyboard.press(Keys.SPRINT.value)
        keyboard.press(Keys.RIGHT.value)
        keyboard.release(Keys.LEFT.value)
        keyboard.release(Keys.JUMP.value)

    @staticmethod
    def backward():
        keyboard.press(Keys.SPRINT.value)
        keyboard.press(Keys.LEFT.value)
        keyboard.release(Keys.RIGHT.value)
        keyboard.release(Keys.JUMP.value)

    @staticmethod
    def jump(height):
        keyboard.press(Keys.JUMP.value)
        time.sleep(height)
        keyboard.release(Keys.JUMP.value)
        keyboard.release(Keys.RIGHT.value)
        keyboard.release(Keys.LEFT.value)

    @staticmethod
    def jump_forward(height):
        keyboard.press(Keys.SPRINT.value)
        keyboard.press(Keys.RIGHT.value)
        keyboard.press(Keys.JUMP.value)
        time.sleep(height)
        keyboard.release(Keys.JUMP.value)
        keyboard.release(Keys.LEFT.value)

    @staticmethod
    def jump_backward(height):
        keyboard.press(Keys.SPRINT.value)
        keyboard.press(Keys.LEFT.value)
        keyboard.press(Keys.JUMP.value)
        time.sleep(height)
        keyboard.release(Keys.JUMP.value)
        keyboard.release(Keys.RIGHT.value)


    @staticmethod
    def stay():
        keyboard.release(Keys.RIGHT.value)
        keyboard.release(Keys.JUMP.value)
        keyboard.release(Keys.LEFT.value)

    @staticmethod
    def release_all_keys():
        """
        Releases all the keys, used before and after playing so that all
        keys are not being pressed
        """
        keyboard.release(Keys.LEFT.value)
        keyboard.release(Keys.UP.value)
        keyboard.release(Keys.DOWN.value)
        keyboard.release(Keys.RIGHT.value)
        keyboard.release(Keys.JUMP.value)
        keyboard.release(Keys.RESET.value)
        keyboard.release(Keys.SPRINT.value)

    def reset(self):
        keyboard.press(Keys.RESET.value)
        time.sleep(0.5)
