import numpy as np
import random as rd
from random import randint
import matplotlib.pyplot as plt
import pandas as pd
import timeit
import time
import sys
import os


def readFile(fileName, totalItems):
    data = pd.read_csv(fileName)
    valueWeightRatio = []
    itemWeights = data['weight'].tolist()
    itemValues = data['value'].tolist()

    itemWeights = itemWeights[:totalItems]
    itemValues = itemValues[:totalItems]

    for index in range(totalItems):
        valueWeightRatio.append(itemValues[index] / itemWeights[index])

    return itemValues, itemWeights, valueWeightRatio


def dynamicProgram(totalItems, knapsackMaxCapacity, values, weights):
    start = time.time()
    sack = [[0 for x in range(knapsackMaxCapacity + 1)] for x in
             range(totalItems + 1)]

    for p in range(totalItems + 1):
        for q in range(knapsackMaxCapacity + 1):
            if p == 0 or q == 0:
                sack[p][q] = 0
            elif weights[p - 1] <= q:
                sack[p][q] = max(
                    values[p - 1] + sack[p - 1][q - weights[p - 1]],
                    sack[p - 1][q])
            else:
                sack[p][q] = sack[p - 1][q]
    stop = time.time()
    dp_exe_time = stop - start

    return sack[totalItems][knapsackMaxCapacity], dp_exe_time


def bruteForce(knapsackMaxCapacity, weights, values, totalItems) -> int:
    if totalItems == 0 or knapsackMaxCapacity == 0:
        return 0

    if weights[totalItems - 1] > knapsackMaxCapacity:
        return bruteForce(knapsackMaxCapacity, weights, values,
                             totalItems - 1)
    else:
        value = max(values[totalItems - 1] + bruteForce(
            knapsackMaxCapacity - weights[totalItems - 1], weights, values,
            totalItems - 1),
                    bruteForce(knapsackMaxCapacity, weights, values,
                                  totalItems - 1))
    return value


def greedyAlgo(knapsackCapacity, itemValues, itemWeights,
               valueWeightRatio):
    start = time.time()
    knapsack = []
    selected_weights = []
    selected_values = []
    knapsackWeight = 0
    knapsackValue = 0

    while knapsackWeight <= knapsackCapacity:
        maxItem = max(valueWeightRatio)
        indexOfMaxItem = valueWeightRatio.index(maxItem)
        if itemWeights[indexOfMaxItem] + knapsackWeight <= knapsackCapacity:
            selected_weights.append(itemWeights[indexOfMaxItem])
            selected_values.append(itemValues[indexOfMaxItem])
            knapsack.append(indexOfMaxItem + 1)
            knapsackWeight += itemWeights[indexOfMaxItem]
            knapsackValue += itemValues[indexOfMaxItem]
            valueWeightRatio[indexOfMaxItem] = -1
        else:
            break

    stop = time.time()
    greedy_exe_time = stop - start

    return knapsackValue, selected_weights, selected_values, greedy_exe_time


def printValue(value):
    print("Value of items in the Knapsack =", value)


def printAllValues(dpValue, dp_exe_time, bruteValue, brute_exe_time, greedyValue, greedy_exe_time):
    print("Time taken for Greedy =", greedy_exe_time, " ms")
    print("Value of items for Greedy =", greedyValue)
    print("Time taken for Bruteforce =", brute_exe_time, " ms")
    print("Value of items for Bruteforce =", bruteValue)
    print("Time taken for Dynamic =", dp_exe_time, " ms")
    print("Value of items for Dynamic =", dpValue)


def calcFitness(weight, value, population, threshold):
    fitness = np.empty(population.shape[0])
    for i in range(population.shape[0]):
        sum1 = np.sum(population[i] * value)
        sum2 = np.sum(population[i] * weight)
        if sum2 <= threshold:
            fitness[i] = sum1
        else:
            fitness[i] = 0

    return fitness.astype(int)


def selectionProcess(fitness, num_parents, population):
    fitness = list(fitness)
    parents = np.empty((num_parents, population.shape[1]))
    for i in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))
        parents[i, :] = population[max_fitness_idx[0][0], :]
        fitness[max_fitness_idx[0][0]] = -999999

    return parents


def crossOver(parents, num_offsprings):
    offsprings = np.empty((num_offsprings, parents.shape[1]))
    crossover_point = int(parents.shape[1]/2)
    crossover_rate = 0.8
    a = 0
    while parents.shape[0] < num_offsprings:
        x = rd.random()
        if x > crossover_rate:
            continue
        parent1_index = a % parents.shape[0]
        parent2_index = (a+1) % parents.shape[0]
        offsprings[a, 0:crossover_point] = parents[parent1_index, 0:crossover_point]
        offsprings[a, crossover_point:] = parents[parent2_index, crossover_point:]
        a += 1

    return offsprings


def mutationProcess(offsprings):
    mutants = np.empty(offsprings.shape)
    mutation_rate = 0.5
    for b in range(mutants.shape[0]):
        random_value = rd.random()
        mutants[b, :] = offsprings[b, :]
        if random_value > mutation_rate:
            continue
        int_random_value = randint(0, offsprings.shape[1]-1)
        if mutants[b, int_random_value] == 0:
            mutants[b, int_random_value] = 1
        else:
            mutants[b, int_random_value] = 0

    return mutants


def optimizerFunction(weight, value, population, pop_size, num_generations, threshold):
    parameters, fitness_history = [], []
    num_parents = int(pop_size[0]/2)
    num_offsprings = pop_size[0] - num_parents
    for c in range(num_generations):
        fitness = calcFitness(weight, value, population, threshold)
        fitness_history.append(fitness)
        parents = selectionProcess(fitness, num_parents, population)
        offsprings = crossOver(parents, num_offsprings)
        mutants = mutationProcess(offsprings)
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = mutants
    print('Final Population: \n{}'.format(population))
    fitness_last_gen = calcFitness(weight, value, population, threshold)
    print('Fitness of the Last Generation: \n{}'.format(fitness_last_gen))
    max_fitness = np.where(fitness_last_gen == np.max(fitness_last_gen))
    parameters.append(population[max_fitness[0][0],:])

    return parameters, fitness_history


def genetic(totalItems, itemWeights, itemValues, knapsackMaxCapacity):
    start = time.time()
    item_number = np.arange(1, totalItems+1)
    weight = itemWeights
    value = itemValues
    knapsack_threshold = knapsackMaxCapacity
    if totalItems >= 32:
        solutions_per_pop = totalItems
    else:
        solutions_per_pop = totalItems * 2

    pop_size = (solutions_per_pop, item_number.shape[0])
    print('Population size = {}'.format(pop_size))
    initial_population = np.random.randint(2, size=pop_size)
    initial_population = initial_population.astype(int)
    if totalItems > 100:
        num_generations = 100
    else:
        num_generations = 50

    print('Initial Popultaion: \n{}'.format(initial_population))
    parameters, fitness_history = optimizerFunction(weight, value, initial_population, pop_size, num_generations, knapsack_threshold)
    print('Optimum parameters for input data are: \n{}\n'.format(str(parameters)[7:-2]))
    selected_items = item_number * parameters

    knapsackValue = 0
    for d in range(selected_items.shape[1]):
        if selected_items[0][d] != 0:
            knapsackValue += itemValues[selected_items[0][d]-1]

    # Fitness Graph
    fitness_history_mean = [np.mean(fitness) for fitness in fitness_history]
    fitness_history_max = [np.max(fitness) for fitness in fitness_history]
    plt.plot(list(range(num_generations)), fitness_history_mean, label='Mean Fitness')
    plt.plot(list(range(num_generations)), fitness_history_max, label='Max Fitness')
    plt.legend()
    plt.title('Fitness Over Generations')
    plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.show()
    stop = time.time()
    print("Time taken for Genetic =", (stop - start), "ms")
    print("Value of items for Genetic =", knapsackValue)
    genetic_exe_time = stop - start

    return genetic_exe_time


# Start Execution
print("\nSolving the 0/1 Knapsack Problem..")

# Input file path
fileName = input("Enter path to Dataset: ")
fd = open(fileName, "r")
d = fd.read()
fd.close()
m = d.split("\n")
s = "\n".join(m[:])
fd = open(fileName, "w+")
for i in range(len(s)):
    fd.write(s[i])
fd.close()

# Convert input file to csv
fin = open(fileName, "rt")
if os.path.exists("csv.csv"):
    os.remove("csv.csv")
fout = open("csv.csv", "wt")
data = fin.read()
data = data[:-1]
data = data.replace(' ', ',')
data = "weight,value\n" + data
fout.write(data)
fin.close()
fout.close()
fileName = 'csv.csv'

# Input # of items in Knapsack
setValue = True
knapsackMaxCapacity = 1000
totalItems = 102
while totalItems >= 101:
    totalItems = int(input("Enter # of items to be read (<=100): "))
    if totalItems > 100:
        print("Error! You can only select upto 100 items.")

# Load Knapsack Parameters
itemValues, itemWeights, valueWeightRatio = readFile(fileName,
                                                            totalItems)

# Print list of items
item_number = np.arange(1, totalItems+1)
print('\nItems read from Dataset:- ')
print('Item #   Weight   Value')
for i in range(item_number.shape[0]):
    print('{0}          {1}         {2}\n'.format(item_number[i], itemWeights[
        i],
                                                  itemValues[i]))

# Plot Weights vs Values of read items
xs = itemWeights
ys = itemValues
plt.scatter(xs, ys, color='w', edgecolor='green', linewidth=1.5, s=100)

plt.title("Weights & Values Distribution")
plt.xlabel("Weights")
plt.ylabel("Values")
plt.show()

# Algorithm Menu
while setValue:
    print("Select Algorithm to be executed:")
    print("1: Brute Force\n2: Dynamic Programming\n3: Greedy\n4: Genetic\n5: "
          "All Algorithms")
    option = input("Enter your option #: ")

    if option == '1':
        print("\nExecuting Brute Force:-")
        start = timeit.default_timer()
        # Execute Brute Force
        bruteValue = bruteForce(knapsackMaxCapacity, itemValues,
                                   itemWeights, totalItems)
        stop = timeit.default_timer()
        # Print Brute Force Execution Time
        print("Time taken for Brute Force =", (stop - start), "ms")
        printValue(bruteValue)
        setValue = False
        break

    if option == '2':
        print("\nExecuting Dynamic Programming:-")
        # Execute DP
        dpValue, dp_exe_time = dynamicProgram(totalItems, knapsackMaxCapacity,
                                  itemValues, itemWeights)
        # Print DP Execution Time
        print("Time taken for Dynamic Programming =", dp_exe_time, "ms")
        printValue(dpValue)
        setValue = False
        break

    if option == '3':
        print("\nExecuting Greedy Approach:-")
        # Execute Greedy
        greedyValue, selected_weights, selected_values, greedy_exe_time = \
            greedyAlgo(knapsackMaxCapacity, itemValues, itemWeights,
                       valueWeightRatio)
        # Print Greedy Execution Time
        print("Time taken for Greedy =", greedy_exe_time, "ms")
        printValue(greedyValue)
        setValue = False
        break

    if option == '4':
        print("\nExecuting Genetic Algorithm:-")
        genetic_exe_time = genetic(totalItems, itemWeights, itemValues,
                            knapsackMaxCapacity)
        setValue = False
        break

    if option == '5':
        print("\nExecuting All Algorithms:-")

        # Execute Dynamic Programming
        dpValue, dp_exe_time = dynamicProgram(totalItems, knapsackMaxCapacity,
                                  itemValues, itemWeights)

        # Execute Brute Force
        start = time.time()
        bruteValue = bruteForce(knapsackMaxCapacity, itemWeights,
                                   itemValues, totalItems)
        stop = time.time()
        brute_exe_time = stop - start

        # Execute Greedy
        greedyValue, selected_weights, selected_values, greedy_exe_time = \
            greedyAlgo(knapsackMaxCapacity, itemValues, itemWeights,
                       valueWeightRatio)

        # Execute Genetic Algorithm
        genetic_exe_time = genetic(totalItems, itemWeights, itemValues,
                                   knapsackMaxCapacity)

        # Printing All Values and Execution Times
        printAllValues(dpValue, dp_exe_time, bruteValue, brute_exe_time, greedyValue, greedy_exe_time)

        # Graph of All Execution Times
        ax = plt.subplot(111)
        ax.set_xlim(-0.2, 3.2)
        ax.grid(b=True, which='major', color='k', linestyle=':', lw=.5, zorder=1)

        x = np.arange(4)
        y = np.array([float(str(genetic_exe_time)[:6]),float(str(greedy_exe_time)[:6]),
                      float(str(brute_exe_time)[:6]),float(str(dp_exe_time)[:6])])

        up = max(y) * .03
        ax.set_ylim(0, max(y) + 3 * up)

        ax.bar(x, y, align='center', width=0.2, color='g', zorder=4)

        for xi, yi, l in zip(*[x, y, list(map(str, y))]):
            ax.text(xi - len(l) * .02, yi + up, l,
                    bbox=dict(facecolor='w', edgecolor='w', alpha=.5))
        ax.set_xticks(x)
        ax.set_xticklabels(['genetic', 'greedy', 'brute', 'dynamic'])
        ax.set_xlabel("Algorithm", fontsize=14)
        ax.set_ylabel("Time in ms", fontsize=14)
        ax.set_title("Execution Time", fontsize=18)
        ax.tick_params(axis='x', which='major', labelsize=12)
        plt.show()
        setValue = False
        break

'''    # Repeat execution with different Algorithm
    print("\nRepeat execution with different Algorithm?")
    while True:
        option2 = input("Press 'y' or 'n' : ")
        if option2 == 'n':
            print("Terminating Program..\nGoodbye!")
            setValue = False
            break
        elif option2 == 'y':
            break
        else:
            print("Error! Please enter a valid option.")'''

