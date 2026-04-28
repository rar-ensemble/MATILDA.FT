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

    float *fr, *d_fr;       // [M*Dim] real-space gradient potential
    float *fx, *d_fx;       // [M] real-space gradient in -x
    float *fy, *d_fy;       // [M] real-space gradient in -y
    float *fz, *d_fz;       // [M] real-space gradient in -z (if DIm > 2)
};



#endif