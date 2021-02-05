import numpy as np
import cv2
from PIL import ImageGrab as im
from PIL import Image
import time
import sys, string, os
from directkeys import PressKey, ReleaseKey, W, A, S, D, JUMP, RUN, L, ENTER
import random
import math
from neural_network import Mario
import os
import pickle
from screeninfo import get_monitors
import pyautogui as pag

POPULATION_ITERATIONS = 500
FIRST_LAYER_SIZE = 100
HIDDEN_LAYER_ONE_SIZE = 64
HIDDEN_LAYER_TWO_SIZE = 16
OUTPUT_LAYER_SIZE = 6
MUTATION_SELECTION_CHANCE = 0.1 # chance that a mario will be selected for mutation
POPULATION_SIZE = 100
START_WAIT_TIME = 3


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
        population = self.create_population(self.population_size)
        for iteration in range(POPULATION_ITERATIONS):
            print("PLAYING MARIO")
            population = self.play_mario(population)
            population.sort(key=lambda Mario: Mario.fitness) # sort population by fitness
            self.save_mario(population[0], "mario" + str(iteration))
            population = self.evolve_mario(population)
            print(f"THE BEST MARIO SO FAR HAS A FITNESS FUNCTION OF {population[0].fitness}")

    def evolve_mario(self, population):

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
            # This part isn't working rn
        return population

    @staticmethod
    def release_all_keys():
        """
        Releases all the keys, used before and after playing so that all
        keys are not being pressed
        """
        # ReleaseKey(A)
        # ReleaseKey(W)
        # ReleaseKey(S)
        # ReleaseKey(D)
        # ReleaseKey(JUMP)
        # ReleaseKey(L)
        # PressKey(RUN)
        pag.keyUp('a')
        pag.keyUp('w')
        pag.keyUp('s')
        pag.keyUp('d')
        pag.keyUp('space')
        pag.keyUp('l')
        pag.keyUp('shiftleft')
        pag.keyUp('enter')


    @staticmethod
    def reset():
        pag.keyDown('l')

    def play_mario(self, population):
        """
        Takes a list of NeuralNetwork objects and uses it to play mario.
        Each NeuralNetwork will play one life and its performance will determine
        what its fitness function will be.
        """
        # need to read in the game screen
        prev_screen = im.grab(bbox=(0, 0, self.x_size, self.y_size))
        prev_screen = cv2.cvtColor(np.array(prev_screen), cv2.COLOR_RGB2GRAY) # converts screen to gray so that the screen array is smaller
        for i, mario in enumerate(population):
            self.reset()
            self.release_all_keys()
            # play the game
            mario.alive = True
            mario.num_times_min_exceded = 0
            print(f"\tmario {i} is starting")
            start_time = time.time()
            while mario.alive:
                curr_screen = im.grab(bbox=(0, 0, self.x_size, self.y_size))
                curr_screen = cv2.cvtColor(np.array(curr_screen), cv2.COLOR_RGB2GRAY)
                mario.play(curr_screen.flatten()) # this won't work right now lol
                mario.update(prev_screen, curr_screen)
                prev_screen = curr_screen
                end_time = time.time()
                start_time = end_time
            self.release_all_keys()
            print(f"\tmario {i} has died he had a fitness function of {mario.fitness}\n")
        return population

    def save_mario(self, mario, filename):
        with open(filename, 'wb') as output:  # Overwrites any existing file.
            pickle.dump(mario, output, pickle.HIGHEST_PROTOCOL)

    def load_mario(self, filename):
        with open(filename, 'rb') as mario_brain:
             mario = pickle.load(mario_brain)
             return mario

# def crossover(mario1, mario2):
#     weights1 = mario1.model.get_weights()
#     print(weights1)
#     weights2 = mario2.model.get_weights()
#     print(weights2)



if __name__ == "__main__":
    time.sleep(START_WAIT_TIME)
    my_mario = MarioGame()
    my_mario.learn_to_play_mario()
