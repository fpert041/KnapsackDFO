# DB-DFO
Dynamic Binary Dispersive Flies Optimisation implemented in C++


### Copyright (C) 2017 Francesco Perticarari and Jasper Kirton-Wingate, with precious advice from Bashar Saade

#### License: This is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License.

[link to the original paper by Mohammad Majid al-Rifaie](http://doc.gold.ac.uk/mohammad/DFO/)

## Results:

We've compared the results of our algorithm against benchmark standards as provided by previous refeence papers. In particular we have compared our results with those obtained by [Salman et alia](http://www.ijmlc.org/vol6/586-L043.pdf) and by [Lopez](https://link.springer.com/article/10.1007/s00500-017-2511-0)

Below you can see a table that compares the results of our DB-DFO algorithm against its original version (DFO) and the current state of the art in both Swarm Intelligence (PSO-based algorithms) and Evolutionary Computation (DE-based algorithms).
![Alt text](dynamic-binary-dispersive.jpg?raw=true "Results")

## What is DFO?

Dispersive Flies Optimisation Research (DFO) [1] is a swarm intelligence algorithm. That is, it is an algorithm where individual agents follow simple instructions to search for a solution in in a matrix representing an N-dimensional search space [2]. The agents are set up so that they can communicate with one another and adapt their search trajectory based on the behaviour of their neighbours.

The algorithm aims to exploit the emergent adaptive intelligence of the swarm to find an optimal solution to a given search problem. Its behaviour is inspired by the swarming behaviour of flies hovering over food sources and it has been shown to find optimal or “good enough” solutions [3] faster than randomness and of the standard versions of the well-known Particle Swarm Optimisation, Genetic Algorithm (GA) as well as Differential Evolution (DE) algorithms on an extended set of benchmarks over three performance measures of error, efficiency and reliability [4].

The algorithm was first proposed by Mohammad Majid al-Rifaie, a computing lecturer at Goldsmiths, University of London, in his paper published in 2014. For a more detailed description, please see [this blog post](http://francescoperticarari.com/dispersive-flies-optimisation/) on my website.

## DB-DFO

The first modification we made, which optimised the algorithm for binary problems, was to constrain the flies positions from floating points values to whole values at the end of the update process.

The next major step in our modification was to reduce the dimensionality of the problem for a number of iterations where we mapped 4 consecutive dimensions onto one number between 0 and 8. This in general reduces the complexity of the search space and allows for position updates for each agent that are not necessarily linear. This speeds up initial calculations, increases the influence of the dispersion threshold, and thus increases the exploration of the search space. We found that having the dimensionality reduction as an initial phase and then 'switching' back to binary vectors of 1s and 0s of dimensions, was much more effective in finding the Global Optima.

#### Visualisation of the Algorithm in action:

![alt text](https://i1.wp.com/upload.wikimedia.org/wikipedia/commons/e/ec/ParticleSwarmArrowsAnimation.gif "Visualisation of the Algorithm in action")

## Knapsack Problem and Technical Notes

The algorithm is hereby used to solve the (multiple) Knapsack allocation problem as described on [this website](http://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/knapsack_01.html).


## How to use the DFO source-code

The algorithm itself is contained in the DFO_cpp folder, which can be extracted and placed into any C++ project.

To use the DFO in a different program you need to create an instance of the algorithm. The wrapper class is "DFO", so just link `DFO.h` to your program and make sure the source files are part of the bindings of your program. Once you have done this, you have to create the DFO instance by calling one of its constructors, then you can use:

```
generateSwarm();
```

-- to generate a new swarm and initialise its parameters (you can edit those in `GlobalParam.hpp`)

```
generateSwarmPositiveAxis();
```
-- ALTERNATIVE: to generate the swarm so that it starts off only on the positive hyperoctant of our search space (hyperoctane = n-dimensional version of a quadrant)

```
updateSwarm();
```

-- to run a cycle of the algorithm, evaluate the current positions of the flies and update the swarm based on the flies' interactions  (you can edit the settings of the algorithm in `GlobalParam.hpp`)

#### Note: the default DFO() constructor is set up with a test function!
#### TO HAVE YOUR OWN FITNESS FUNCTION: You must pass in a function (normal, lambda or using a std::function type varibale as an argument for the constructor)

## Other useful parameters you can set:

* dim: the dimensions of the problem (default = 10)
 `GlobalParam::dim `
* popSize: the size of the population of 'flies' (default = 20)
 `GlobalParam::popSize `
* disturbance threshold (default to = 0.001)
 `GlobalParam::dt `
* Constant to set the maximum number of Fly Evaluations allowed to the program (default = 300000 - useflu to avoid infinite loops)
 `GlobalParam::FE_allowed `
 * boolean that sets whether the algorithm will make the individuals optimise only according to their "best neighbour" (true) or according to the "best in the swarm" too (false = default)
  `GlobalParam::democracy `
  * boolean that sets whether the flies will be constrained within a bounded search space  = it keeps each fly's coordinates within the given search space width
  `DFO::constrainPositions `

There are "setter" and "getter" methods  you can use to change these and other parameters (including setting/changing a custom fitness function). Here are the most useful:

`DFO dfo` or `DFO * pDfo = new DFO()` or  `unique_ptr<DFO> dfo = unique_ptr<DFO>(new DFO())` etc.

-- create a default instance of a DFO "swarm" and its algorithm

`setPopSize( int new_pop_size )`

-- set the population size

`setDemocracy(bool true_or_false )`

-- set the algorithm version

`setDim( int new_dimensions_number )`

-- set the dimensions of the search space ( which is equal to the length of a fly's positional vector, also defineable as its hypothesis or "proposed solution")

`setConstrainPos(bool true_or_false)`

-- set whther files will be allowed to explore an unbounded search space after their initialisation (false = default) or whther their deimensions will be bound to a maximum value equal to the initial search space width and a minimum value of 0

`setSearchSpaceWidth( int new_max_range_for_this_dim )`

-- set the search space positive range of all the dimensions equal to a given input

`setSearchSpaceWidth( int which_dim_to_set , int new_max_range_for_this_dim )`

-- set the search space positive range of a specific dimension equal to a given input

`setFitnessFunc( [this](std::vector<double> p) { fitness = 0; /* DO SOMETHING */ return fitness} )`

-- set sustom fitness function (you can also do this at the start by passing in a function into the DFO constructor)

`setNeighbourTopology(NeighbouringTopologyType nt);`

-- set the neighbouring topology that DFO uses to connect flies (defaults is DFO::RING topolgy), but can be set to DFO::RANDOM

`setDtRandMode(DtRanMode drm)`

-- set the type of randomness governing the disturbance (defaults is DFO::UNI = uniform), but can be set to DFO::GAUSS

`setLeader(std::vector<double> newLeaderPos)`

-- set the leader of the swarm externally using a vector of doubles

`setEvalsCounter(int newEC)`

 -- set counter of calls to evaluation function (evaluation purposes)

 - - -

 `void setNumNeighbours(int num);`

 -- set number of fly's neighbours to be checked per side (defaults is 1 => 1 Left & 1 Right)

 `std::vector<std::shared_ptr<Fly>> getSwarm();`

  -- return the vector of shared_pointers for the swarm

 `std::shared_ptr<Fly> getBestFly();`

 -- return the shared_pointers to the best Fly in the swarm

 `std::vector<double> getSearchSpaceWidth()`

  -- return the search space dimensions vecor

 `int getSearchSpaceWidth(int dim);`

 -- return one specific dimension width

 `int getDim()`

 -- return the number of dimensions

  `int GlobalParam::getNumNeighbours() `

  -- return number of neighbours that each fly checks to find the best neighbour

 `std::string getEvalFuncName()`

   -- return whther the fitness function is a default one (if yes, which one?) or a custom one

 `int getEvalCount()`

 -- return the current cycle of the algorithm

 `int getBestIndex()`

 -- return the swarm's best index

 `double Fly.getFitness()`

 -- Fly's method that returns its fitness

 `str getNeighbourTopology()`

 -- get the neighbouring topology used to link flies

 `float getDt()`

 -- get the disturbance threshold

 `std::string getDtRandMode()`

 -- get the type of randomness governing the disturbance
 
 `int getEvalsCounter()`
 
  -- get counter of calls to evaluation function
 
