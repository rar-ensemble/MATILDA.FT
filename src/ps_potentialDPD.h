// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "ps_potential.h"
#include "ps_neighborList.h"

#ifndef _POTENTIAL_DPD
#define _POTENTIAL_DPD

__global__ void d_DPD_forces(
    float*       d_f,
    const float* d_v,
    const float* d_x,
    const float* d_L,
    const float* d_Lh,
    const int*   d_neighborList,
    const int*   d_nNeighbors,
    const int*   d_siteList,
    curandState* d_states,
    float        gamma,
    float        sigma,
    float        rcut,
    float        inv_sqrt_delt,
    int          maxNeighbors,
    int          nsites,
    int          Dim);

class DPD : public PS_Potential {
public:
    DPD(std::istringstream&, PS_Box*);
    ~DPD() {}
    void initializePotential() override;
    void CalcForces() override;
    float CalcEnergy() override;

    float gamma;
    float kT;
    float sigma;
    float delt;
    PS_NeighborList* nList;
};

#endif
