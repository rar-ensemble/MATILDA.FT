// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "ps_potential.h"

#ifndef _NBCHARGE
#define _NBCHARGE

// Sets up potential for either NP-NP or NP-Gaussian potential
// See manual for complete details
class NBCharge : public PS_Potential {
    protected:

    public:
        NBCharge();      // Default constructor
        NBCharge(std::istringstream&, PS_Box*);  // Actual used constructor
        ~NBCharge();     // Default destructor

        void initializePotential(void) override;
        void CalcForces(void) override;
        float CalcEnergy(void) override;
        
        float LB;           // Bjerrum length
        float sig2;         // Variance for the charge-smearing Gaussian

};

#endif