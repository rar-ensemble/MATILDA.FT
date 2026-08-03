// Copyright (c) 2025 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "ps_potential.h"

#ifndef _PWALL
#define _PWALL


// One-sided harmonic wall acting on a single group.
// Planar mode (axis = x/y/z): U_i = (k/2) * (wallPos - x_i,n)^2 when particle is on the forbidden side; 0 otherwise.
// Radial mode (axis = r/R): same law with x_i,n replaced by the particle's radial distance r from
// the box center (L/2 per dimension); wallPos is then a radius rather than a coordinate.
class Wall : public PS_Potential {

public:
    Wall();
    Wall(std::istringstream&, PS_Box*);
    ~Wall();
    void initializePotential(void) override;
    void CalcForces(void) override;
    float CalcEnergy(void) override;

    int normalDim;      // 0=x, 1=y, 2=z (ignored when isRadial)
    bool isRadial = false; // true when axis was given as r/R
    float wallPos;      // plane coordinate along normal, or radius from box center when isRadial
    int dirSign;        // +1: allowed side is x_n > wallPos (or r > wallPos); -1: opposite side
    float k;            // harmonic stiffness

    float* d_ener;      // per-particle energy buffer (size = group nsites)
};


#endif
