import copy
import math
import random
from operator import attrgetter
import matplotlib.pyplot as plt
from datetime import datetime
import os
from sklearn.model_selection import train_test_split
import time

experimentTitle = "Dataset 3 8 nodes old pop gen ratio"

dataSource = "data/12102638.txt"
HidNODES = 8 # don't forget to modify when changing datasets
P = 50  # stick with these (20/80)
NUMBEROFGENERATIONS = 200
NUMBEROFITERATIONS = 3
GENEMIN = -4
GENEMAX = 4
MUTATIONRATE = 0.2
MUTATIONSTEP = 0.8
crossover = False

bestFitnesses = []
meanFitnesses = []
validationFitnesses = []
generations = []
bestFitnessesOverAllGens = []
totalMeanCalculationList = []

def setNumberOfInputNodes():
    dataFile = open(dataSource, "r")
    fileLine = dataFile.readline()
    lineList = fileLine.split()
    nodeCount = 0
    for i in range(0, len(lineList) - 1):
        nodeCount = nodeCount + 1
        if (fileLine[i] == 1) or (fileLine[i] == 0):
            break
    return nodeCount

InpNODES = setNumberOfInputNodes()
OutNODES = 1
N = ((InpNODES + 1) * HidNODES) + ((HidNODES + 1) * OutNODES)
HNodeOUT = [0] * ((HidNODES + 1) * OutNODES)  # stores the hidden node outputs
INodeOUT = [0] * ((InpNODES + 1) * HidNODES) # stores the output node outputs #  This is our result

# Object with gene and fitness parameters

class Individual:
    def __init__(self):
        self.chromosome = [0]*N # @ USE THIS NOTATION FOR CREATING THE 2D WEIGHT ARRAYS
        self.fitness = 0

# Create an object containing a list of input values and one classification float value

class TestDatum:
    def __init__(self):
        self.input = [0] * InpNODES # cant be inpNodes. data size
        self.expectedResult = 0

# Create an object containing a list of hidden weights, a list of output weights and a value to track error

class Network:
    def __init__(self):
        # hweights is a 2D array with HidNODES rows and InpNODES + 1 columns
        self.hweights = [[0 for i in range(InpNODES + 1)] for j in range (HidNODES)]
        # oweights is a 2D array with OutNODES rows and HidNODES+1 columns
        self.oweights = [[0 for i in range(HidNODES + 1)] for j in range (InpNODES)]
        # this is the error resulting from running the network with the given weights
        self.error = 0  # float

# Perform the sigmoid function

def sigmoid(node):
    if node >= 0:
        z = math.exp(-node)
        sig = 1 / (1 + z)
        return sig
    else:
        z = math.exp(node)
        sig = z / (1 + z)
        return sig

# Function that reads data from an input file and stores it in an object, returning a list of those objects

def extractData():
    file = open(dataSource, "r")
    lines = file.readlines()
    listOfTestData = []
    
    for i in range(0, len(lines)): # For each line in the data file
        lines[i] = lines[i].replace("\n", "").split(' ')
        object = TestDatum()
        for j in range(0, InpNODES):
            object.input[j] = float(lines[i][j])
        object.expectedResult = float(lines[i][InpNODES]) 
        listOfTestData.append(object)

    return listOfTestData

# Assign individual gene values to a neural network's weights

def populateNeuralNetwork(individual):
    network = Network()
    count = 0
    for i in range(HidNODES):
        for j in range(InpNODES + 1):
            network.hweights[i][j] = individual.chromosome[count]
            count += 1
    for i in range(OutNODES):
        for j in range(HidNODES + 1):
            network.oweights[i][j] = individual.chromosome[count]
            count += 1
    return network

# Use evaluate the fitness of an individual using a neural network's classification error value as the fitness value

def calculateFitnessWithNeuralNetwork(individual, dataSet):
    network = populateNeuralNetwork(individual)

    for t in range(0, len(dataSet)):  # Iterate over each object extracted from the input data
        for currentHiddenNode in range(0, HidNODES):
            HNodeOUT[currentHiddenNode] = 0
            for currentInputNode in range(0, InpNODES):
                HNodeOUT[currentHiddenNode] += (network.hweights[currentHiddenNode][currentInputNode] * dataSet[t].input[currentInputNode])
            HNodeOUT[currentHiddenNode] += network.hweights[currentHiddenNode][InpNODES]  # Adds Bias to sum
            HNodeOUT[currentHiddenNode] = sigmoid(HNodeOUT[currentHiddenNode])

        for currentOutputNode in range(0, OutNODES):
            INodeOUT[currentOutputNode] = 0
            for currentHiddenNode in range(0, HidNODES):
                INodeOUT[currentOutputNode] += (network.oweights[currentOutputNode][currentHiddenNode] * HNodeOUT[currentHiddenNode])
            INodeOUT[currentOutputNode] += network.oweights[currentOutputNode][HidNODES] # Bias
            INodeOUT[currentOutputNode] = sigmoid(INodeOUT[currentOutputNode])
    
        if dataSet[t].expectedResult == 1.0 and INodeOUT[0] < 0.5:
            network.error += 1.0  # Class 1 if output > 0.5
        if dataSet[t].expectedResult == 0.0 and INodeOUT[0] >= 0.5:
            network.error += 1.0 # Does this mean one error each go?

    return 100 * (network.error/len(dataSet)) # Added to return percentage

# Evaluate fitness of each individual in a population using the fitness function.

def evaluatePopulation(population, dataSet):
    for individual in population:
        individual.fitness = calculateFitnessWithNeuralNetwork(individual, dataSet)

# Evaluate the fitness of each individual in a population, and then return the sum of those fitnesses

def calculateTotalFitness(population, dataSet):
    evaluatePopulation(population, dataSet)
    populationFitness = 0
    for individual in population:
        populationFitness += individual.fitness
    return populationFitness

# Return the individual in a population with the lowest fitness using the min function on the attrgetter operator.

def returnBestIndividual(population): 
    return min(population, key=attrgetter('fitness'))

# Find the weakest individual in a population and return its index

def returnIndexOfWorstIndividual(population):
    return population.index(max(population, key=attrgetter('fitness')))

# Return the fitness of the individual in a population with the lowest fitness using the min function on the attrgetter operator.

def returnMinimumFitness(population):
    return returnBestIndividual(population).fitness

# Return the mean fitness of individuals in a population.

def returnMeanFitness(population, dataSet):
    totalFitness = calculateTotalFitness(population, dataSet)
    meanFitnesses = totalFitness/len(population)
    return meanFitnesses

# Instantiate a population by adding a gene populated with binary values to an individual object, and adding it to a list

def initialisePopulation():
    population = []
    for x in range(0, P):
        tempChromosome = []
        for y in range(0, N):
            tempChromosome.append(random.uniform(GENEMIN, GENEMAX))
        newInd = Individual()
        newInd.chromosome = tempChromosome.copy() # gene should be named genome
        population.append(newInd)
    return population

# Tournament Selection. Create offspring population by choosing a pair of random parent individuals, and adding the fittest of the two offspring created to a new list. (Selection Code)

def performTournamentSelection(population):
    offspring = []
    for i in range(0, P):
        # Choose two parents from the population, and create an offspring from each.
        parent1 = random.randint(0, P-1)
        off1 = copy.deepcopy(population[parent1])
        parent2 = random.randint(0, P-1)
        off2 = copy.deepcopy(population[parent2])
        # Add the least fit offspring to a new list.
        if off1.fitness < off2.fitness:
            offspring.append(off1)
        else:
            offspring.append(off2)
    # print("Most fit - ", calculateTotalFitness(offspring))
    return offspring

# Roulette wheel selection

# For each pair of objects in offspring, select a crossover point and perform crossover, replacing the original individuals. Now depending on crossover probability.

def performCrossover(population, dataSet):
    toff1 = Individual()
    toff2 = Individual()
    temp = Individual()
    for i in range(0, P, 2):
        toff1 = copy.deepcopy(population[i])
        toff2 = copy.deepcopy(population[i+1])
        temp = copy.deepcopy(population[i])
        crosspoint = random.randint(1, N)
        # At the crossover point, perform crossover on the pair and replace the originals.
        for j in range(crosspoint, N):
            toff1.chromosome[j] = toff2.chromosome[j]
            toff2.chromosome[j] = temp.chromosome[j]
        if (calculateFitnessWithNeuralNetwork(toff1, dataSet) < population[i].fitness):
            population[i] = copy.deepcopy(toff1)
        if (calculateFitnessWithNeuralNetwork(toff2, dataSet) < population[i+1].fitness):
            population[i] = copy.deepcopy(toff2)
    # print("Crossover - ", calculateTotalFitness(population))
    return population

# Flip values in individual's gene strings depending on the mutation probability, and replace it.

def performMutation(population):
    for i in range(0, P):
        newInd = Individual()
        newInd.chromosome = []
        for j in range(0, N):
            gene = population[i].chromosome[j]
            mutationProbability = random.random()
            if mutationProbability < MUTATIONRATE:
                alteration = random.uniform(-MUTATIONSTEP, MUTATIONSTEP)
                gene = gene + alteration
                if gene > GENEMAX:
                    gene = GENEMAX
                elif gene < GENEMIN:
                    gene = GENEMIN
            newInd.chromosome.append(gene)
        population[i] = copy.deepcopy(newInd)
    # print("Mutation - ", calculateTotalFitness(population))
    return population

# Take the weakest individual in a population and place it in the place of the fittest individual in its offspring population

def performElitism(parentPopulation, offspringPopulation):
    weakestIndividualInParentPop = min(
        parentPopulation, key=attrgetter('fitness'))
    indexOfFittestIndividualInOffspringPop = returnIndexOfWorstIndividual(
        offspringPopulation)
    offspringPopulation[indexOfFittestIndividualInOffspringPop] = weakestIndividualInParentPop

# Save mean and best values to lists for plotting purposes

def saveGenerationInfo(population, dataSet):
    meanFitnesses.append(returnMeanFitness(population, dataSet))
    bestFitnesses.append(returnMinimumFitness(population))

# Function containing all the code needed to create the matplotlib plot

def savePlot(iteration):
    bestFitnessLabel = "Best Fitness "
    meanFitnessLabel = "Mean fitness"
    validationFitnessLabel = "Validation Fitness"
    plt.step(bestFitnesses, '', label=bestFitnessLabel)
    plt.plot(meanFitnesses, '', label=meanFitnessLabel)
    plt.plot(validationFitnesses, '', label=validationFitnessLabel)
    plt.xlim(0, NUMBEROFGENERATIONS)
    plt.xlabel("Generations")
    plt.ylim(0, 100)
    plt.ylabel("Fitness")
    plt.legend()
    plt.title("Iteration" + str(iteration + 1))
    plt.savefig(path + "Iteration_" + str(iteration + 1) + "_" + str(int(min(bestFitnesses))) + ".png")
    plt.clf()

# What the tin says

def createAlgorithmInfo():
    return ("Experiment Title: " + experimentTitle + "\n"
        + "Input Nodes: " + str(InpNODES) + "\n"
        + "Input Node Outputs: " + str(len(INodeOUT)) + "\n"
        + "Hidden Nodes: " + str(HidNODES) + "\n"
        + "Hidden Node Outputs: " + str(len(HNodeOUT)) + "\n"
        + "Output Node: " + str(OutNODES) + "\n"
        + "Candidate Solutions (P): " + str(P) + "\n"
        + "Number of Generations: " + str(NUMBEROFGENERATIONS) + "\n"
        + "Number of Iterations: " + str(NUMBEROFITERATIONS) + "\n"
        + "Genome Length (N): " + str(N) + "\n"
        + "Gene Min: " + str(GENEMIN) + "\n"
        + "Gene Max: " + str(GENEMAX) + "\n"
        + "Mutation Rate: " + str(MUTATIONRATE) + "\n"
        + "Mutation Step: " + str(MUTATIONSTEP) + "\n"
        + "Crossover: " + str(crossover) + "\n"
        + "Lowest Fitness: " + str(min(bestFitnessesOverAllGens)) + "%\n"
        + "Mean Lowest Fitness: " + str(round(sum(totalMeanCalculationList) / len(totalMeanCalculationList), 2)) + "%\n"
        + "Mean of Mean Fitnesses: " + str(round(sum(meanFitnesses) / len(meanFitnesses), 2)) + "%\n"
        + "Lowest Validation Fitness: " + str(round(min(validationFitnesses))) + "%\n"
        + "Elapsed time in seconds: " + str(round(time.time() - startTime, 2))) # Turn into hh:mm:ss

# Save data from a run to a file

def saveAlgorithmInfo():
    f = open(path + "Incarnation" + datetime.now().strftime("%m-%d_%H_%M_%S") + ".txt","w+")
    f.write(createAlgorithmInfo())
    f.close()

# Print data from a run to a file

def printAlgorithmInfo():
    print(createAlgorithmInfo())

# Start recording duration

startTime = time.time()

# Choose where and how to save algorithm information

path = os.path.join("Plots/", "plot_" + datetime.now().strftime("%m-%d_%H_%M_%S")+ experimentTitle + "/")
os.mkdir(path)

# Extract and split data # I could be doing this inside the iteration loop
dataList = extractData()
trainingData, testingData = train_test_split(dataList, train_size=0.7)

for iteration in range(NUMBEROFITERATIONS): # Stochastic check

    bestFitnesses.clear()
    meanFitnesses.clear()
    validationFitnesses.clear()

    # TRAINING

    # Initialise a population of random individuals containing network weight genomes
    population = initialisePopulation()

    # Try the neural network with these weights, and assign each individual a fitness.

    evaluatePopulation(population, trainingData)
    saveGenerationInfo(population, trainingData)      

    for gen in range(NUMBEROFGENERATIONS):
        # Select the best individuals in the population.
        offspring = performTournamentSelection(population)
        if crossover == True: performCrossover(offspring, trainingData)
        # Mutate the individuals
        performMutation(offspring)
        # run the network with the new weights
        evaluatePopulation(offspring, trainingData)
        # Replace the worst individual in this generation with the best from the last one
        performElitism(population, offspring)

        # VALIDATION # Any reason why I can't keep this in this loop? It's using a different data set # I should refactor this to be inside saveGenerationInfo
        bestIndividualInGeneration = returnBestIndividual(population)
        fitnessOfBestOnValidationData = calculateFitnessWithNeuralNetwork(bestIndividualInGeneration, testingData)
        validationFitnesses.append(fitnessOfBestOnValidationData)

        population = copy.deepcopy(offspring)

        # Save information from this generation
        saveGenerationInfo(population, trainingData)
        print(min(bestFitnesses))
        bestFitnessesOverAllGens.append(min(bestFitnesses))

    totalMeanCalculationList.append(int(min(bestFitnesses)))
    savePlot(iteration)

saveAlgorithmInfo()
printAlgorithmInfo()
