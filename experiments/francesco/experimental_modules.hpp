//
//  experimental_modules.hpp
//  imgSymmetry
//
//  Created by Francesco Perticarari on 13/12/2017.
//

#ifndef experimental_modules_hpp
#define experimental_modules_hpp

#include <stdio.h>
#include "../../DFO_cpp/include/Utilis.hpp"
#include "../../DFO_cpp/include/helpers.hpp"

using namespace std;

class DFOx : public Utilis {
protected:
    
    bool constrainPositions = false;
    
public:
    
    // default constructor
    DFOx();
    // overloaded constructor where a function is passed in as an argument
    DFOx(std::function<double(std::vector<double>)> fitnessFunc);
    
    DFOx( const DFOx &obj);
    
    // default destructor
    ~DFOx();
    
    // generate DFOx swarm (cycle 0 of the algorithm)
    void const generateSwarm();
    
    // generate DFOx swarm (cycle 0 of the algorithm) using only the positive axis of each dimensions
    void const generateSwarmPositiveAxis();
    
    // DFOx implementation part: evaluate flies, make them interact & update the swarm
    void const updateSwarm();
    
    // keep fly's coordinates within the given search space width
    void const setConstrainPos(bool status);
};

#endif /* experimental_modules_hpp */
