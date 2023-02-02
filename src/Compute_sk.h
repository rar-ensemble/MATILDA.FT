// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "Compute.h"
#include <string>
#include <sstream>
#include <iostream>
#include <complex>
#include <cufft.h>
#include <cufftXt.h>

void write_kspace_data(const char*, std::complex<float>*);
__global__ void d_prepareDensity(int, float*, cufftComplex*, int);
__global__ void d_multiplyComplex(cufftComplex*, cufftComplex*,
    cufftComplex*, int);

#ifndef _COMPUTE_S_K_H_
#define _COMPUTE_S_K_H_

class Sk: public Compute {

protected:
    int particle_type;
    void writeResults_instantly();

public:

    void allocStorage() override;
    void doCompute(void);
    void writeResults();
    Sk(std::istringstream&);
    ~Sk();

};

#endif // _COMPUTE_S_K_H_
