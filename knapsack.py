#DFO for the knapsack adapted from Leon Feddens python DFO code
import numpy as np
import math

# Parse our data from .txt
# Read all the data from the file
readFile = open('mknap.txt','r').readlines()
params = readFile[0]
#print(yes.split())

# Number of bags
nBags = 30
nBags = int(params[1:3])
#print(nBags)

# Dimension number, number items
totalI = int(params[4:6])
#print(totalI)

readVal = []
valuesList = []  # Values the same size as items
lines = int(totalI/10)
for i in range(1, lines+1):
   readVal.append(readFile[i])

values = []
for i in range(len(readVal)):
    valuesList.append(readVal[i].strip('\n'))
    values.append(list(map(int, valuesList[i].split())))

values = np.array(values).flatten()

lines = int(nBags/10)
readCon = []
cons = []
for i in range(7, lines+7):
    readCon.append(readFile[i])

# Our maximum weight capacity
max_weight = []
for i in range(len(readCon)):
    cons.append(readCon[i].strip('\n'))
    max_weight.append(list(map(int, cons[i].split())))

max_weight = np.array(max_weight).flatten()
#print(npCons)

readWeights = []
weightList = []
lines = int((nBags*totalI)/10)
for i in range(10, lines+10):
    readWeights.append(readFile[i])
# 30 x 60 constraints
weights = []
for i in range(len(readWeights)):
    weightList.append(readWeights[i].strip('\n'))
    weights.append(list(map(int, weightList[i].split())))

weights = np.array(weights).flatten()
weights = np.split(weights, nBags)

# end read data


# Let's print the output nicely
np.set_printoptions(precision=3, suppress=True)

# Population Size
size = 100

# How regularly the flies are dispersed
disturbance_threshold = 0.1

def fitness(i): # Fitness for each fly
    fly_fitness = 0
    sum_weights = []
    np.around(population[i][:], 0, roundedP)  # Rounded fly values
    roundedP.astype(int)
    #print(roundedP)
    sum_values = np.sum((roundedP[:]) * (values[:]))  # Values are always the same for each bag
    fly_fitness = sum_values
    for n in range(0, nBags):  # For each bag, check the weights
        sum_weights.append(np.sum((roundedP[:])*(weights[n][:])))  # Constraints are different for each bag
        if sum_weights[n] > max_weight[n]:  # If sum of the weights for that bag > the relevant capacity
            fly_fitness = 0

    return fly_fitness  # Fitness is always a reflection of the values (change this?)


#target values (ideal solution) = 7772

# Create a solution population; include certain items
population = np.array([np.ones(totalI) for _ in range(size)])

# initial solution generation
for i in range(len(population)):
    for j in range(0, totalI):
        population[i][j] = np.random.uniform(0.0, 1.0)

# to store our rounded binary int population
roundedP = np.zeros(totalI)


# Because the algorithm is stochastic sometimes it is nice to record the best solution 
all_time_best = np.zeros_like(population[0])
all_time_best_score = 0

# An empty vessel to contain each flies best neighbour
best_neighbour = np.zeros_like(population[0])


# How many iterations of optimisation we want to compute
iteration_amount = 1000  # Passed profit = 6000 for the first time with 1000 iterations

divisor = 15  # Our new number of dimensions
chunks = totalI//divisor
print(chunks)

for _ in range(iteration_amount):
    # Compute the fitnesses for each fly
    fitnesses = np.zeros(len(population))
    for i in range(len(population)):
        fitnesses[i] = fitness(i)

    # Which flies index has the highest value?
    swarms_best_index = np.argmax(fitnesses)

    # Get best fly
    swarms_best = population[swarms_best_index]

    # Record best fly of all time
    if np.amax(fitnesses) >= all_time_best_score:
        all_time_best_score = np.amax(fitnesses)
        np.around(swarms_best, 0, all_time_best)
        #all_time_best = swarms_best

    # All the random dice rolls we will make for every 'dimension' in each member of the population with a normal distribution
    dr = np.random.normal(0.0, 1.0, divisor)

    # For each fly in the swarm
    for i, p in enumerate(population):

        # Get the neighbours indices - note how we turn the list into a circular buffer for the ring topology
        left = (i - 1) if i is not 0 else len(population) - 1
        right = (i + 1) if i is not (len(population) - 1) else 0

        # Here is the best scoring neighbouring fly
        best_neighbour = population[left] if fitnesses[left] > fitnesses[right] else population[right]

        # For each element comprising the fly
        # Could I bypass this with dimensionality reduction?

        #p = np.array([1, 0, 1, 0, 1, 1, 1, 0])

        ld = []
        ld = np.split(p, divisor)

        for j in range(0, divisor): # Store a new array of less dimensions than totalI and update on new dims
            ld[j] = ld[j].dot(1 << np.arange(ld[j].size)[::-1])  # bit shift conversion of float binary values
            #print(ld[j])

            # If the roll computed earlier is lower than the threshold, re-init the fly, else, update
            # fly to best neighbour and move it a random amount towards the swarms best fly.
            if dr[j] < disturbance_threshold:
                ld[j] = np.random.uniform(0.0, math.pow(2, chunks)-2)  # Max value for binary -> decimal (itemsI / divisor) | restrict search space?
                #ld[j] = np.random.uniform(0.0, math.pow(2, (totalI//divisor)))
            else:
                update = swarms_best[j] - best_neighbour[j]
                #update = best_neighbour[j]
                ld[j] = np.random.uniform(0.0, 1.0) * update

            # convert back to our original dimensions to get fitnesses
            ld[j] = int(np.round(ld[j], decimals=0))
            #ld[j] = math.floor(ld[j])
            # binary dims for every decimal dim after update
            p[(j*chunks):(j+1)*chunks] = np.fromstring(np.binary_repr(ld[j], width=chunks), dtype='S1').astype(int)
        #print(p)

# Get the final fitnesses
fitnesses = np.zeros(len(population))
for i in range(len(population)):
    fitnesses[i] = fitness(i)

# Get the best fly
swarms_best_index = np.argmax(fitnesses)
swarms_best_value = np.amax(fitnesses)
swarms_best = population[swarms_best_index]

print(fitnesses)
print(swarms_best_index)
print('best profit ever:  ', all_time_best_score)
print('best fly ever:  ', all_time_best)
#print('c best profit: ', swarms_best_value)
#print('c best fly: ', swarms_best)