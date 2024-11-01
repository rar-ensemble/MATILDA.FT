// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "ps_potential.h"

#ifndef _NBGAUSS
#define _NBGAUSS

class NBGauss : public PS_Potential {
    protected:

    public:
        NBGauss();      // Default constructor
        NBGauss(std::istringstream&, PS_Box*);  // Actual used constructor
        ~NBGauss();     // Default destructor
        void initializePotential(void);
        
        float Ao;           // Gaussian potential prefactor
        float sig2;         // Variance for the Gaussian

        std::string grpI, grpJ; // Groups on which this potential acts
        int Iind, Jind;     // Group indices on which this potential acts
};

#endif