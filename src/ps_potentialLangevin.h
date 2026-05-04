// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "ps_potential.h"

#ifndef _PLANGEVIN
#define _PLANGEVIN


// This class is of type Potential and inherits all Potential
// routines. Potential is constructed first, then the instance-specific
// constructor routines called.
class Langevin : public PS_Potential {

public:
    Langevin();
    Langevin(std::istringstream&, PS_Box*);
    ~Langevin();
    void initializePotential(void) override;
    void CalcForces(void) override;
    float CalcEnergy(void) override;


    float drag;         // friction coefficient
    float delt;         // time step used
    float noise_mag;    // Magnitude of random noise
};



#endif