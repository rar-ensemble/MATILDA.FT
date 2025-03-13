// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "ps_potential.h"

#ifndef _NBERF2
#define _NBERF2

class NBErf2 : public PS_Potential {
    protected:

    public:
        NBErf2();      // Default constructor
        NBErf2(std::istringstream&, PS_Box*);  // Actual used constructor
        ~NBErf2();     // Default destructor
        void initializePotential(void) override;
        
        float Ao;       // Gaussian potential prefactor
        float Rp1, Rp2; // Particle radius
        float xi1, xi2; // Variance for the Gaussian
};

#endif