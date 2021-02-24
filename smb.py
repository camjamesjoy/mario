import numpy as np
import cv2
import argparse

from PIL import ImageGrab as im
from PIL import Image
import time
import sys, string, os
import random
import math
from neural_network import Mario
import os
import pickle
from screeninfo import get_monitors
import keyboard
from multiprocessing import Process, Pipe

POPULATION_ITERATIONS = 500
FIRST_LAYER_SIZE = 100
HIDDEN_LAYER_ONE_SIZE = 64
HIDDEN_LAYER_TWO_SIZE = 16
OUTPUT_LAYER_SIZE = 4
MUTATION_CHANCE = 0.1 # chance that a mario will be selected for mutation
POPULATION_SIZE = 25
START_WAIT_TIME = 3
 # How often the fitness function will be updated in seconds
SAVED_MARIO_FOLDER = "saved_marios"

class MarioGame:
    """
    A class that holds information about the mario game.
    """
    def __init__(self):
        monitor = get_monitors()[0] # grabs the first moniter
        self.x_size = monitor.x # x size of first monitor, will be used to read game screen
        self.y_size = monitor.height # y size of first monitor, will be used to read game screen
        self.screen_size = self.x_size * self.y_size - 1
        self.population_size = POPULATION_SIZE


    def create_population(self, size):
        """
        Creates a list of NeuralNetwork objects that are empty.
        These are the objects that will be trained to play mario.
        The returned NeuralNetworks will not yet have a fitness function assigned
        to them
        """
        population = []
        hidden_layers = []
        for _ in range(size):
            population.append(Mario(FIRST_LAYER_SIZE, OUTPUT_LAYER_SIZE, hidden_layers, self.screen_size))
        return population

    def learn_to_play_mario(self):
        population = self.create_population(self.population_size)
        for iteration in range(POPULATION_ITERATIONS):
            print("PLAYING MARIO")
            population = self.play_mario(population)
            population.sort(key=lambda Mario: Mario.fitness) # sort population by fitness
            self.save_mario(population[-1], f"{SAVED_MARIO_FOLDER}\\mario_{str(iteration)}_{str(int(population[-1].fitness))}")
            print(f"THE BEST MARIO SO FAR HAS A FITNESS FUNCTION OF {population[-1].fitness}")
            population = self.evolve_mario(population)

    def evolve_mario(self, population):

        population = population[self.population_size // 2 : self.population_size] # sve the best half of marios
        best = population[len(population) - 1] # don't want to mutate the best one
        population = population + self.create_population(self.population_size // 2)

        for mario in population:
            if random.random() < MUTATION_CHANCE and mario is not best:
                # mario has been selected for mutation
                mario.mutate()
                print("\t\t\t WHOA MARIO IS EVOLVING")
            if random.random() < MUTATION_CHANCE and mario is not best:
                # Inputs selected for mutation
                mario.mutate_input_data()
                print("\t\t\t WHOA MARIO's INPUTS ARE EVOLVING")
        return population

    def play_mario(self, population):
        """
        Takes a list of NeuralNetwork objects and uses it to play mario.
        Each NeuralNetwork will play one life and its performance will determine
        what its fitness function will be.
        """
        # need to read in the game screen
        for i, mario in enumerate(population):
            mario.reset()
            mario.release_all_keys()
            # play the game
            mario.alive = True
            mario.fitness = 0
            mario.num_times_min_exceded = 0
            print(f"\tmario {i} is starting")
            start = time.time()
            parent_conn, child_conn = Pipe()
            fitness_process = Process(target=mario.update, args = (0,0,self.x_size,self.y_size, child_conn))
            fitness_process.start()
            while mario.alive:
                curr_screen = im.grab(bbox=(0, 0, self.x_size, self.y_size))
                curr_screen = cv2.cvtColor(np.array(curr_screen), cv2.COLOR_RGB2GRAY)
                mario.play(curr_screen.flatten())
                if parent_conn.poll():
                    update = parent_conn.recv()
                    mario.alive = update[0]
                    mario.fitness = update[1]
                if keyboard.is_pressed("Esc"):
                    print("escape key pressed. exiting")
                    mario.release_all_keys()
                    fitness_process.join()
                    sys.exit(0)

            mario.release_all_keys()
            fitness_process.join()
            print(f"\tmario {i} has died he had a fitness function of {mario.fitness}\n")
        return population

    def save_mario(self, mario, filename):
        with open(filename, "wb") as output:  # Overwrites any existing file.
            pickle.dump(mario, output, pickle.HIGHEST_PROTOCOL)

    def load_mario(self, filename):
        with open(filename, "rb") as mario_brain:
             mario = pickle.load(mario_brain)
             return mario

# def crossover(mario1, mario2):
#     weights1 = mario1.model.get_weights()
#     print(weights1)
#     weights2 = mario2.model.get_weights()
#     print(weights2)



if __name__ == "__main__":
    time.sleep(START_WAIT_TIME)
    parser = argparse.ArgumentParser(description="Play Mario!")
    # parser.add_argument("--verbose", "-v", action="store_true", required=False)
    parser.add_argument("--mario", "-m", required=False, nargs='+', help="Path to a saved mario, program will load that mario and play it then exit")
    args = parser.parse_args()
    marios = args.mario
    if marios:
        for saved_mario in marios:
            mario = MarioGame()
            mario_brain = mario.load_mario(saved_mario)
            mario.play_mario([mario_brain])
        quit()
    my_mario = MarioGame()
    my_mario.learn_to_play_mario()
