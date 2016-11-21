# -*- coding: utf-8 -*-
## ©Shamir Alavi, Nov 20, 2016

import random
import matplotlib.pyplot as plt
import numpy as np
import time

from deap import base
from deap import creator
from deap import tools

numCities = 51
coordinates = [[37,52],[49,49],[52,64],[20,26],[40,30],[21,47],[17,63],[31,62],[52,33],[51,21],[42,41],[31,32],[5,25],[12,42],
               [36,16],[52,41],[27,23],[17,33],[13,13],[57,58],[62,42],[42,57],[16,57],[8,52],[7,38],[27,68],[30,48],[43,67],
               [58,48],[58,27],[37,69],[38,46],[46,10],[61,33],[62,63],[63,69],[32,22],[45,35],[59,15],[5,6],[10,17],[21,10],
               [5,64],[30,15],[39,10],[32,39],[25,32],[25,55],[48,28],[56,37],[30,40]]
cities = range(51)
cityDict = zip(cities, coordinates)             # bounds each city with its coordinates, (sort of) a look up table

average = []
minFit = []

creator.create("minTourLength", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.minTourLength)

toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(0,numCities), numCities)   # random sampling
toolbox.register("individual", tools.initIterate, creator.Individual,
                           toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Evaluation heuristic/ Fitness function
def evaluateSample(numIterations, originalDict, sample):
    # returns the sum of euclidean distances
    # input parameters: numCities is the number of cities (integer)
    #                   originalDict is the look up table containing
    #                   the city IDs and their corresponding coordinates (2-D list)
    #                   sample is a random sample of the population (1-D list)
    # output: sum of euclidean distances (float)
    # limitation: does not bound check on the iterable that is passed into the function (currently hardcoded)
    eucl_distance = []
    temp = 0
    for i in xrange(numIterations-1):
        currentCityCoord = np.asarray(originalDict[sample[i]][1])
        nextCityCoord = np.asarray(originalDict[sample[i+1]][1])
        temp = np.linalg.norm(currentCityCoord - nextCityCoord)   # superfast, at least 2x faster than euclidean()
        eucl_distance.append(temp)
    return [sum(eucl_distance)]

# Create toolbox functions for our GA
toolbox.register("evaluate", evaluateSample, numCities, cityDict)
toolbox.register("mate", tools.cxOrdered)    # crossover
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.02)    # Indexes are shuffled for mutation
toolbox.register("select", tools.selTournament, tournsize = 3 )

def main():
    
    start = time.clock()
    random.seed(46)
    
    Population = toolbox.population(n=int(numCities*1.5))
    
    CXPB, MUTPB, NGEN = 0.9, 0.2, 1       # probability of crossover and mutation; number of generations to evolve
    
    print("Start of evolution")
    
    # Evaluate the entire population
    fitnesses = map(toolbox.evaluate, Population)
    
    k = 750         # selection parameter
    for g in range(NGEN):
        if ((g % 100 == 0) or g == (NGEN - 1)):
            print("-- Generation %i --" % g)
                  
        # Select the next generation individuals
        offspring = toolbox.select(Population, k)
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
    
        # Apply mutation on the offspring
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            
        #print("  Evaluated %i individuals" % len(invalid_ind))
        
        # The population is entirely replaced by the offspring
        Population[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in Population]
        
        length = len(Population)
        mean = sum(fits) / length
        #sum2 = sum(x*x for x in fits)
        #std = abs(sum2 / length - mean**2)**0.5
        
        #print("  Min %s" % min(fits))
        #print("  Max %s" % max(fits))
        #print("  Avg %s" % mean)
        #print("  Std %s" % std)
        
        average.append(mean)
        minFit.append(min(fits))
    
    print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(Population, 1)[0]
    print "-------  Statistics  -------"
    print "Generation number: ", g
    print "best tour length: ", best_ind.fitness.values[0]
    print("Best tour: %s" % (best_ind))
    print "Average tour length: ", np.mean(np.asarray(average))
    print "_____________________________________"
    print "Parameters:"
    print "Fitness heuristic: minimum distance (Euclidean)"
    print "Population size = ", int(numCities*1.5)
    print "Crossover probability = ", CXPB, ", ", "type: Tournament Selection (Tournament size = 3, k = ", k, ")"
    print "Mutation probability = ", MUTPB, ", ", "type: Shuffle Indexes (shuffle probability = 0.02)"
    
    fig, ax1 = plt.subplots()
    ax1.plot(range(len(average)), average, "b-", label="Average Tour Length")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Average Tour Length", color="b")
    plt.show()
    
    end = time.clock()
    timeElapsed = end - start
    print "timeElapsed = ", timeElapsed, " seconds"
    print "\n"
    
if __name__ == "__main__":
    main()