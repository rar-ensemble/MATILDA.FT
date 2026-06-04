// Copyright (c) 2025 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "ps_potential.h"

#ifndef _PWALL
#define _PWALL


// One-sided axis-aligned harmonic wall acting on a single group.
// U_i = (k/2) * (wallPos - x_i,n)^2 when particle is on the forbidden side; 0 otherwise.
class Wall : public PS_Potential {

public:
    Wall();
    Wall(std::istringstream&, PS_Box*);
    ~Wall();
    void initializePotential(void) override;
    void CalcForces(void) override;
    float CalcEnergy(void) override;

    int normalDim;      // 0=x, 1=y, 2=z
    float wallPos;      // plane coordinate along normal
    int dirSign;        // +1: allowed side is x_n > wallPos; -1: allowed side is x_n < wallPos
    float k;            // harmonic stiffness

    float* d_ener;      // per-particle energy buffer (size = group nsites)
};


#endif
