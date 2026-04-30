// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "ps_integrator.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <sstream>


__global__ void d_EM_integrator(float*, const float*, const float, const float,
const int*, const float*, const int*, const int, const int, curandState*);


#ifndef _INTEGRATOR_EM_
#define _INTEGRATOR_EM_

class PS_Box;

class EM : public Integrator {
public:
    EM(std::istringstream&, PS_Box*);
    ~EM();
    void Integrate_2() override;
	void finishInitialization() override;

};

#endif
