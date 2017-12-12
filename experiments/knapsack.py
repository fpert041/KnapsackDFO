#DFO for the knapsack adapted from Leon Feddens python DFO code
#num of dimensions = 10 (i think)
import numpy as np

#making the solution vector of floats

# Let's print the output nicely
np.set_printoptions(precision=3, suppress=True)


# Dimension size, number items
totalI = 24

# Population Size
size = totalI*4

# How regularily the flies are dispersed
disturbance_threshold = 0.15

# our maximum weight capacity
max_weight = 6404180


weights = np.array([382745,
                    799601,
                    909247,
                    729069,
                    467902,
                    44328,
                    34610,
                    698150,
                    823460,
                    903959,
                    853665,
                    551830,
                    610856,
                    670702,
                    488960,
                    951111,
                    323046,
                    446298,
                    931161,
                    31385,
                    496951,
                    264724,
                    224916,
                    169684])

values = np.array([825594,
                   1677009,
                   1676628,
                   1523970,
                   943972,
                   97426,
                   69666,
                   1296457,
                   1679693,
                   1902996,
                   1844992,
                   1049289,
                   1252836,
                   1319836,
                   953277,
                   2067538,
                   675367,
                   853655,
                   1826027,
                   65731,
                   901489,
                   577243,
                   466257,
                   369261])

#target values (ideal solution) = 309

# Create a solution population; include certain items
population = np.array([np.ones(totalI) for _ in range(size)])

# initial solution generation
for i in range(len(population)):
    for j in range(0, totalI):
        population[i][j] = np.random.uniform(0.0, 1.0)

# to store our binary int population
roundedP = np.zeros(totalI)

def fitness(i):
    fitness = 0
    np.around(population[i][:], 0, roundedP)
    #print(roundedP)
    sumWeights = np.sum((roundedP[:])*(weights[:]))
    sumValues = np.sum((roundedP[:])*(values[:]))
    if (sumWeights > max_weight):
        fitness = 0
    else:
        fitness = sumValues
    return fitness


# Because the algorithm is stochastic sometimes it is nice to record the best solution 
all_time_best = np.zeros_like(population[0])
all_time_best_score = 0

# An empty vessel to contain each flies best neighbour
best_neighbour = np.zeros_like(population[0])



# How many iterations of optimisation we want to compute
iteration_amount = 10000

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

    # All the random dice rolls we will make for every 'dimension' in each member of the population with a normal distribution
    dr = np.random.normal(0.0, 1.0, population.shape)

    # For each fly in the swarm
    for i, p in enumerate(population):

        # Get the neighbours indices - note how we turn the list into a circular buffer for the ring topology
        left = (i - 1) if i is not 0 else len(population) - 1
        right = (i + 1) if i is not (len(population) - 1) else 0

        # Here is the best scoring neighbouring fly
        best_neighbour = population[left] if fitnesses[left] > fitnesses[right] else population[right]

        # For each element comprising the fly
        for x in range(len(p)):
            # If the roll computed earlier is lower than the threshold, re-init the fly, else, update
            # fly to best neighbour and move it a random amount towards the swarms best fly.
            if dr[i][x] < disturbance_threshold:
                p[x] = np.random.uniform(0.0, 1.0)
            else:
                update = swarms_best[x] - best_neighbour[x]
                p[x] = best_neighbour[x] + np.random.uniform(0.0, 1.0) * update

# Get the final fitnesses
fitnesses = np.zeros(len(population))
for i in range(len(population)):
    fitnesses[i] = fitness(i)

# Get the best fly
swarms_best_index = np.argmax(fitnesses)
swarms_best = population[swarms_best_index]

print(fitnesses)
print(swarms_best_index)
print('best profit:  ', all_time_best_score)
print('best fly ever:  ', all_time_best)
