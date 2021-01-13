import numpy as np
import cv2
from PIL import ImageGrab as im
from PIL import Image
import time
import sys, string, os
from directkeys import PressKey, ReleaseKey, W, A, S, D, JUMP, RUN, L
import random
import math
from neural_network import Mario
import os
import pickle
from screeninfo import get_monitors


POPULATION_ITERATIONS = 500
FIRST_LAYER_SIZE = 100
HIDDEN_LAYER_ONE_SIZE = 64
HIDDEN_LAYER_TWO_SIZE = 16
OUTPUT_LAYER_SIZE = 6
MUTATION_SELECTION_CHANCE = 0.1 # chance that a mario will be selected for mutation
POPULATION_SIZE = 100


class MarioGame:
    """
    A class that holds information about the mario game.
    """
    def __init__(self):
        monitor = get_monitors()[0] # grabs the first moniter
        self.x_size = monitor.x # x size of first monitor, will be used to read game screen
        self.y_size = monitor.height # y size of first monitor, will be used to read game screen
        self.population_size = POPULATION_SIZE


    def create_population(self, size):
        """
        Creates a list of NeuralNetwork objects that are empty.
        These are the objects that will be trained to play mario.
        The returned NeuralNetworks will not yet have a fitness function assigned
        to them
        """
        population = []
        hidden_layers = [HIDDEN_LAYER_ONE_SIZE, HIDDEN_LAYER_TWO_SIZE]
        for _ in range(size):
            population.append(Mario(FIRST_LAYER_SIZE, OUTPUT_LAYER_SIZE, hidden_layers))
        return population

    def learn_to_play_mario(self):
        population = create_initial_population(self.population_size)
        for iteration in range(POPULATION_ITERATIONS):
            population = play_mario(population)
            population = evolve_mario(population)

    def evolve_mario(self, population):
        population.sort(key=lambda Mario: Mario.fitness) # sort population by fitness
        population = next_population[self.population_size // 2 : self.population_size] # sve the best half of marios
        population = population + self.create_population(population_size // 2)
        best = population[0] # don't want to mutate the best one
        for mario in population:
            if random.random() < MUTATION_CHANCE and mario is not best:
                # mario has been selected for mutation
                mario.mutate()
            if random.random() < MUTATION_CHANCE and mario is not best:
                # Add inputs.
                mario.mutate_input_data()

        return population

    @staticmethod
    def release_all_keys():
        """
        Releases all the keys, used before and after playing so that all
        keys are not being pressed
        """
        ReleaseKey(A)
        ReleaseKey(W)
        ReleaseKey(S)
        ReleaseKey(D)
        ReleaseKey(JUMP)
        PressKey(RUN)

    def play_mario(self, population):
        """
        Takes a list of NeuralNetwork objects and uses it to play mario.
        Each NeuralNetwork will play one life and its performance will determine
        what its fitness function will be.
        """
        # need to read in the game screen
        prev_screen = im.grab(bbox=(0, 0, self.x_size, self.y_size))
        prev_screen = cv2.cvtColor(np.array(prev_screen), cv2.COLOR_RGB2GRAY) # converts screen to gray so that the screen array is smaller
        for mario in population:
            self.release_all_keys()
            # play the game
            mario.alive = True
            mario.num_times_min_exceded = 0
            while mario.alive:
                curr_screen = im.grab(bbox=(0, 0, self.x_size, self.y_size))
                curr_screen = cv2.cvtColor(np.array(curr_screen), cv2.COLOR_RGB2GRAY)
                mario.play(curr_screen.flatten()) # this won't work right now lol
                mario.update(prev_screen, curr_screen)
                prev_screen = curr_screen
            self.release_all_keys()
        return population

    def save_mario(self, mario, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(mario, output, pickle.HIGHEST_PROTOCOL)

    def load_mario(self, filename):
        with open(filename, 'rb') as mario_brain:
             mario = pickle.load(mario_brain)
             return mario

    def align():
        # This part needs to be redone. It is not sufficient to rely on the user to center
        # the screen because it could be off by a bit and when loading the model back up
        # the inputs would not be the same
        # I think the easiest way to get around this is to require the user to play in full screen mode
        monitor = get_monitors()[0] # grabs the first moniter
        x_size = monitor.x # x size of first monitor
        y_size = monitor.height # y size of first monitor

        while(1):
            screen = im.grab(bbox=(0,0,x_size,y_size))
            grayScreen = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2GRAY)
            cv2.imshow("align", np.array(grayScreen))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

def crossover(mario1, mario2):
    weights1 = mario1.model.get_weights()
    print(weights1)
    weights2 = mario2.model.get_weights()
    print(weights2)



if __name__ == "__main__":
    my_mario = MarioGame()
    pop = my_mario.create_population(100)
    pop[0].mutate_input_data()
    my_mario.play_mario(pop)
