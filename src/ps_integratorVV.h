// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "ps_integrator.h"

#ifndef _INTEGRATOR_VV_
#define _INTEGRATOR_VV_

class PS_Box;

__global__ void d_VV_integrate_1(float*, float*, const float*, const float*,
    const int*, const float*, const int*, const int, const int, const float);

__global__ void d_VV_integrate_2(float*, const float*, const float*,
    const int*, const int*, const int, const int, const float);

class VV : public Integrator {
public:
    VV(std::istringstream&, PS_Box*);
    ~VV();
    void Integrate_1() override;
    void Integrate_2() override;
    void finishInitialization() override;

    float* d_invMass;   // [nTypes] 1/mass, precomputed in finishInitialization
};

#endif
