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

#ifndef _COMPUTE_AVG_CHEMPOT_H_
#define _COMPUTE_AVG_CHEMPOT_H_

class ChemPot : public Compute {
private:
    int nmolecules;
    int ntries;
    int number_stored;
protected:
    int first_mol_id;  
    int last_mol_id;
    int ns_per_molec;
    float fraction_to_sample;

public:

    void allocStorage() override;
    void doCompute(void);
    void writeResults();
    ChemPot(std::istringstream&);
    ~ChemPot();

};

#endif