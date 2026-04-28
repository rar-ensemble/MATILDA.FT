// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "ps_potential.h"

#ifndef _NBERFG
#define _NBERFG

class NBErfG : public PS_Potential {
    protected:

    public:
        NBErfG();
        NBErfG(std::istringstream&, PS_Box*);
        ~NBErfG();
        void initializePotential(void) override;

        float Ao;    // potential prefactor
        float Rp;    // step-function radius
        float sig2;  // Gaussian smearing variance (sigma²)
};

#endif
