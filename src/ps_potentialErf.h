// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "ps_potential.h"

#ifndef _NBERF
#define _NBERF

class NBErf : public PS_Potential {
    protected:

    public:
        NBErf();      // Default constructor
        NBErf(std::istringstream&, PS_Box*);  // Actual used constructor
        ~NBErf();     // Default destructor
        void initializePotential(void) override;
        
        float Ao;           // Gaussian potential prefactor
        float Rp;           // Particle radius
        float sigma;         // Variance for the Gaussian
};

#endif