// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "ps_potential.h"

#ifndef _NBERFG
#define _NBERFG

class NBErfG : public PS_Potential {
    protected:

    public:
        NBErfG();      // Default constructor
        NBErfG(std::istringstream&, PS_Box*);  // Actual used constructor
        ~NBErfG();     // Default destructor
        void initializePotential(void) override;
        
        float Ao;   // potential prefactor
        float Rp;   // Particle radius
        float xi;   // Particle interface width
        float sigma;// Variance for the Gaussian
};

#endif