// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "ps_potential.h"

#ifndef _NBBIASFIELD
#define _NBBIASFIELD


// This class is of type Potential and inherits all Potential
// routines. Potential is initialized first, then the Gaussian
// initialization is performed.
class BiasField : public PS_Potential {

public:
    BiasField();
    BiasField(std::istringstream&, PS_Box*);
    ~BiasField();
    void initializePotential(void) override;
    void CalcForces(void) override;
    float CalcEnergy(void) override;

    float Ao;           // Magnitude of the bias field
    int n_periods;      // Number of periods
    int dir;            // Direction (if relevant)
    std::string phase;  // Phase to bias towards

    
};



#endif