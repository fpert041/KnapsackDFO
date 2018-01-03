#DFO for the knapsack adapted from Leon Feddens python DFO code
import numpy as np
import math

# Parse our data from .txt
# Read all the data from the file
readFile = open('sento2.txt', 'r').readlines()
params = readFile[0]
#print(yes.split())
goRaise = False

# Number of bags
#nBags = 30
nBags = int(params[1:3])
print(nBags)

# Dimension number, number items
totalI = int(params[4:6])
print(totalI)

readVal = []
valuesList = []  # Values the same size as items

lines = int(np.round(totalI/10, decimals=0))
for i in range(1, lines+1):
    readVal.append(readFile[i])

values = []
for i in range(len(readVal)):
    valuesList.append(readVal[i].strip('\n'))
    values.append(list(map(int, valuesList[i].split())))

values = np.array(values).flatten()
#values = [item for sublist in values for item in sublist]
#values = np.array(values)
print(values)

lines = int(np.round(nBags/10, decimals=0))
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
print(max_weight)

readWeights = []
weightList = []
lines = int(np.round((nBags*totalI/10), decimals=0))
print(lines)
for i in range(10, lines+10):
    readWeights.append(readFile[i])
# nBags * totalI constraints
weights = []
for i in range(len(readWeights)):
    weightList.append(readWeights[i].strip('\n'))
    weights.append(list(map(int, weightList[i].split())))

weights = np.array(weights).flatten()
#weights = [item for sublist in weights for item in sublist]
#weights = np.array(weights)
#print(weights)
weights = np.split(weights, nBags)

#print(weights)

# end read data


# Let's print the output nicely
np.set_printoptions(precision=3, suppress=True)

# Population Size
size = 50

# How regularly the flies are dispersed initially
disturbance_threshold = 0.01


#maximum_profit_poss = np.sum(values)
#print(maximum_profit_poss)

FE = 0

def fitness(i, alpha): # Fitness for each fly
    global FE
    FE += 1
    sum_weights = []
    np.around(population[i][:], 0, roundedP)  # Rounded fly values
    roundedP.astype(int)
    #print(roundedP)
    sum_values = np.sum((roundedP[:]) * (values[:]))  # Values are always the same for each bag
    fly_fitness = sum_values
    # Perhaps we could have a dynamic fitness function?
    for n in range(0, nBags):  # For each bag, check the weights
        sum_weights.append(np.sum((roundedP[:])*(weights[n][:])))  # Constraints are different for each bag
        if sum_weights[n] > max_weight[n]:  # If sum of the weights for that bag > the relevant capacity
            fly_fitness = fly_fitness*alpha

return fly_fitness  # Minimisation function



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


# Passed profit = 6000 for the first time with 1000 iterations

# number of chunks
#divisor = 14
#chunkSize = totalI//divisor  # Our new number of dimensions for updating
#print(chunkSize)

# How many iterations of optimisation we want to compute
iteration_amount = 0

# Penalisation factor
alpha = 0.7

# Vector of best profits
best_profits = np.zeros(100000)


while True:
    iteration_amount += 1
    # Compute the fitnesses for each fly
    fitnesses = np.zeros(len(population))
    for i in range(len(population)):
        fitnesses[i] = fitness(i, alpha)

    # Which flies index has the highest value?
    swarms_best_index = np.argmax(fitnesses)

# Get best fly
swarms_best = population[swarms_best_index]

# Record best fly of all time
if np.amax(fitnesses) >= all_time_best_score:
    all_time_best_score = np.amax(fitnesses)
    np.around(swarms_best, 0, all_time_best)
    np.append(best_profits, all_time_best_score)
    #all_time_best = swarms_best
    
    # All the random dice rolls we will make for every 'dimension' in each member of the population with a normal distribution or uniform
    #if best_profits[iteration_amount] == best_profits[iteration_amount-1000]:
    #    disturbance_threshold = 0.0
    
    # Can we represent alpha (penalisation) as an oscillation over time? Even the disturbance threshold?
    # We treat FE as time
    t = FE/50
    alpha = abs(math.sin(math.pi*0.106*t)*0.996)
    disturbance_threshold = abs(math.cos(math.pi*0.01*t)*0.08)
    
    #alpha -= 0.0001
    #disturbance_threshold += 0.000001
    #if (np.round(alpha, 2) < 0.99 and goRaise == False):
    # print(np.round(alpha, 2))
    #alpha -= 0.00005
    
    #if np.round(alpha, 2) > 0.99:
    #    goRaise = False
    #    disturbance_threshold -= 0.000001
    
    #elif goRaise:
    #    alpha += 0.00005
    
    #elif np.round(alpha, 2) <= 0.01:
    #    print(np.round(alpha, 2))
    #    goRaise = True
    
    
    dr = np.random.uniform(0.0, 1.0, population.shape)
    
    
    # For each fly in the swarm
    for i, p in enumerate(population):
        
        # Get the neighbours indices - note how we turn the list into a circular buffer for the ring topology
        left = (i - 1) if i is not 0 else len(population) - 1
        right = (i + 1) if i is not (len(population) - 1) else 0
        
        # Here is the best scoring neighbouring fly
        # Search in 4 to the left and to the right to find best
        '''
            if fitnesses[left-3] > fitnesses[right] and fitnesses[right-1] and fitnesses[right-2] and fitnesses[right-3]:
            best_neighbour = population[left-3]
            elif fitnesses[left-3] <= fitnesses[right] and fitnesses[right-1] and fitnesses[right-2] and fitnesses[right-3]:
            best_neighbour = population[right-3]
            
            if fitnesses[left-2] > fitnesses[right] and fitnesses[right-1] and fitnesses[right-2] and fitnesses[right-3]:
            best_neighbour = population[left-2]
            elif fitnesses[left-2] <= fitnesses[right] and fitnesses[right-1] and fitnesses[right-2] and fitnesses[right-3]:
            best_neighbour = population[right-2]
            '''
        if fitnesses[left-1] > fitnesses[right] and fitnesses[right-1] and fitnesses[right-2]:
            best_neighbour = population[left-1]
        elif fitnesses[left-1] <= fitnesses[right] and fitnesses[right-1] and fitnesses[right-2]:
            best_neighbour = population[right-1]
        elif fitnesses[left] > fitnesses[right] and fitnesses[right-1] and fitnesses[right-2]:
            best_neighbour = population[left]
        elif fitnesses[left] <= fitnesses[right] and fitnesses[right-1] and fitnesses[right-2]:
            best_neighbour = population[right]
        
        #best_neighbour = population[left] if fitnesses[left] > fitnesses[right] else population[right]
        
        #best_neighbour = population[left-3] if (fitnesses[left-3] > fitnesses[right] and fitnesses[right-1] and fitnesses[right-2] and fitnesses[right-3]) else population[right-3]
        
        
        
        # For each element comprising the fly
        # Could I bypass this with dimensionality reduction?
        
        #p = np.array([1, 0, 1, 0, 1, 1, 1, 0])
        
        #ld = []
        #ld = np.split(p, divisor)
        
        # For each element comprising the fly
        for x in range(len(p)):
            # If the roll computed earlier is lower than the threshold, re-init the fly, else, update
            # fly to best neighbour and move it a random amount towards the swarms best fly.
            if dr[i][x] < disturbance_threshold:
                p[x] = np.random.uniform(0.0, 1.0)
                p[x] = np.rint(p[x])
            else:
                update = swarms_best[x] - best_neighbour[x]
                p[x] = best_neighbour[x] + np.random.uniform(0.0, 1.0) * update
                p[x] = np.rint(p[x])

#print(p)

# Get the final fitnesses
fitnesses = np.zeros(len(population))
for i in range(len(population)):
    fitnesses[i] = fitness(i, alpha)
    
    # Get the best fly
    swarms_best_index = np.argmax(fitnesses)
    swarms_best_value = np.amax(fitnesses)
    swarms_best = population[swarms_best_index]
    
    best_at_iteration = [0]
    i = 0
    if iteration_amount % 50 == 0:
        best_at_iteration.append(all_time_best_score)
        if best_at_iteration[i] == best_at_iteration[i - 1]:
            disturbance_threshold = disturbance_threshold / 2
        i = i + 1
        print('alpha: ', alpha)
        print('dt: ', disturbance_threshold)
        print('iteration no:  ', iteration_amount)
        print('best profit ever:  ', all_time_best_score)
        print('best solution: ', all_time_best)
        print('FEs: ', FE)
        print(fitnesses)
#print(swarms_best_index)
#print('best fly ever:  ', all_time_best)
#print('c best profit: ', swarms_best_value)
#print('c best fly: ', swarms_best)
