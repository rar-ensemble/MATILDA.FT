// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


//////////////////////////////////////////
// Rob Riggleman            8 July 2022 //
// Class for dealing with potentials    //
// that act on tensor fields. Have the  //
// non-local Maier-Saupe potential in   //
// mind as the first candidate.         //
//////////////////////////////////////////

#ifndef TENSOR_PAIRSTYLE
#define TENSOR_PAIRSTYLE
#include <complex>
#include <cufft.h>
#include <cufftXt.h>

class TensorPotential {
public:

    float *Tfield1, *Tfield2, *d_Tfield1, *d_Tfield2;

    TensorPotential();
    virtual ~TensorPotential();
    std::string PrintCommand();
    void Initialize_TensorPotential();

};
#endif


__global__ void d_multiply_cufftCpx_scalar(cufftComplex*, float, int);