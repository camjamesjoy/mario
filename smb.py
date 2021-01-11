import numpy as np
import cv2
from PIL import ImageGrab as im
from PIL import Image
import time
import sys, string, os
from directkeys import PressKey, ReleaseKey, W, A, S, D, JUMP, RUN, L
import pygame
import random
import math
import neural_network
import os
import pickle
from screeninfo import get_monitors


trainingIterations = 500
firstLayerSize = 4200
secondLayerSize = 64
thirdLayerSize = 16
outputLayerSize = 6
death_lim = 1000

population_size = 100
mutation_chance = 0.1




def main():
    # aligns the board that the program reads in and the game board
    # mario5 = load_mario("mario5")
    # mario6 = neural_network.neural_network(firstLayerSize, secondLayerSize, thirdLayerSize, outputLayerSize, 1,
    #                        mario5.layer1["weights"], mario5.layer2["weights"], mario5.layer3["weights"])
    #
    # play(mario5)
    #
    population = create_initial_pop()
    best = population[0]
    population.sort(key=lambda neural_network: neural_network.fitness)
    next_population = population[math.floor(population_size/2):population_size]
    m = 1
    while(best.fitness < 15000):

        while(len(next_population) < population_size):
            temp = neural_network.neural_network(firstLayerSize, secondLayerSize, thirdLayerSize, outputLayerSize, 1)
            next_population.append(temp)
        for i in range(len(next_population)):
            if(random.random() < mutation_chance):
                new_mario = next_population[i].mutate(firstLayerSize, secondLayerSize, thirdLayerSize, outputLayerSize)
                next_population[i] = new_mario
            play(next_population[i])
            if (next_population[i].fitness > best.fitness):
                best = next_population[i]

        next_population.sort(key=lambda neural_network: neural_network.fitness)
        next_population = next_population[math.floor(population_size/2):population_size]
        print(best.fitness)
        if(best.fitness > 1000*m):
            save_mario(best, "mario" + str(m))
            m += 1


def create_initial_pop():

    population = [None] * population_size
    # global firstLayerSize
    # screen1 = im.grab(bbox=(500,400,800,670))
    # screen1 = screen1.resize((75,56))
    # grayScreen1 = cv2.cvtColor(np.array(screen1), cv2.COLOR_RGB2GRAY)
    # firstLayerSize = len(np.array(grayScreen1).flatten())
    for j in range(population_size):

        population[j] = neural_network.neural_network(firstLayerSize, secondLayerSize, thirdLayerSize, outputLayerSize, 1)
        play(population[j])

    return population

def align():
    # This part needs to be redone. It is not sufficient to rely on the user to center
    # the screen because it could be off by a bit and when loading the model back up
    # the inputs would not be the same
    # I think the easiest way to get around this is to require the user to play in full screen mode
    monitor = get_monitors()[0]
    x_size = monitor.x
    y_size = monitor.height

    while(1):
        screen = im.grab(bbox=(0,0,x_size,y_size))
        # cv2.imshow("align1", np.array(screen))
        # screen = screen.resize((75,56))
        grayScreen = cv2.cvtColor(np.array(screen), cv2.COLOR_RGB2GRAY)
        cv2.imshow("align", np.array(grayScreen))
        #print(len(np.array(grayScreen).flatten()))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

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

def play(mario):

    # releases all keys before starting

    i = 0
    death_counter = 0 # num times mario has died
    screen1 = im.grab(bbox=(500,400,800,670)) #grabs the game screen
    screen1 = screen1.resize((75,56)) # shrinks game screen for faster processing later

    marioAlive = True
    maxFitnessVal = 0
    grayScreen1 = cv2.cvtColor(np.array(screen1), cv2.COLOR_RGB2GRAY)
    while((marioAlive or death_counter <= death_lim)):
        screen2 = im.grab(bbox=(500,400,800,670))
        screen2 = screen2.resize((75,56))
        grayScreen2 = cv2.cvtColor(np.array(screen2), cv2.COLOR_RGB2GRAY)
        mario.play(np.array(np.array(grayScreen2).flatten()))

        # cv2.imshow("test", grayScreen2)
        # cv2.imshow("test2", np.array(grayScreen1))
        if i % 10 == 0:
            grayScreen1 = grayScreen2
        maxFitnessVal, marioAlive = fitness(grayScreen1, grayScreen2, maxFitnessVal)
        if(marioAlive == False):
            death_counter += 1
        i += 1

    mario.fitness = maxFitnessVal
    #print(str(maxFitnessVal))
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(S)
    ReleaseKey(D)
    ReleaseKey(JUMP)
    ReleaseKey(RUN)
    PressKey(L)
    time.sleep(1)
    ReleaseKey(L)


def crossover(mario1, mario2):
    weights1 = mario1.model.get_weights()
    print(weights1)
    weights2 = mario2.model.get_weights()
    print(weights2)

def fitness(screen1, screen2, maxFitnessVal):
    screenDiff = np.mean(np.array(screen2) - np.array(screen1))
    marioAlive = (screenDiff >= 3)
    # print(marioAlive)
    if(marioAlive):
        fitnessVal = maxFitnessVal
        if(screenDiff <= 5):
            fitnessVal -= (screenDiff + 5) * 10
        else:
            fitnessVal += screenDiff
        if(fitnessVal > maxFitnessVal):
            maxFitnessVal = fitnessVal

    return maxFitnessVal, marioAlive

def save_mario(mario, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(mario, output, pickle.HIGHEST_PROTOCOL)

def load_mario(filename):
    with open(filename, 'rb') as mario_brain:
         mario = pickle.load(mario_brain)
         return mario

if __name__ == "__main__":
    #align()
    #main()
