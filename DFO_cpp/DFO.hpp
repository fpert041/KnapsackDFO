//
//  DFO.hpp
//  wk2_DFO
//
//  Created by Francesco Perticarari on 14/10/2017.
//

#ifndef DFO_hpp
#define DFO_hpp

#include <stdio.h>
#include "include/Utilis.hpp"
#include "include/helpers.hpp"

using namespace std;

class DFO : public Utilis {
protected:
    
    bool constrainPositions = false;
    bool binaryProblem = false;
    bool discreteProblem = false;
    bool keepMoving = false;

public:
    
    // default constructor
    DFO();
    // overloaded constructor where a function is passed in as an argument
    DFO(std::function<double(std::vector<double>)> fitnessFunc);

    DFO( const DFO &obj);
    
    // default destructor
    ~DFO();
    
    // generate DFO swarm (cycle 0 of the algorithm)
    void const generateSwarm();
    
    // generate DFO swarm (cycle 0 of the algorithm) using only the positive axis of each dimensions
    void const generateSwarmPositiveAxis();
    
    // DFO implementation part: evaluate flies, make them interact & update the swarm
    void const updateSwarm();
    
    // keep fly's coordinates within the given search space width
    void const setConstrainPos(bool status);
    
    // fly's coordinates are rounded to either 1 or 0 at the end of the update function
    void const isBinaryProblem(bool status);
    
    // fly's coordinates are rounded to either 1 or 0 at the end of the update function
    void const isDiscreteProblem(bool status);
    
    bool const getDiscreteProblem();
    
    void const setkeepMoving(bool status);
    
    bool const getkeepMoving();
};

#endif /* DFO_hpp */
