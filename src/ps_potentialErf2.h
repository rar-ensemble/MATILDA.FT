// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "ps_potential.h"

#ifndef _NBERF2
#define _NBERF2

class NBErf2 : public PS_Potential {
    protected:

    public:
        NBErf2();
        NBErf2(std::istringstream&, PS_Box*);
        ~NBErf2();
        void initializePotential(void) override;

        float Ao;    // potential prefactor
        float Rp;    // step-function radius
        float sig2;  // Gaussian smearing variance (sigma²)
};

#endif
