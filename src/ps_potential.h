// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#ifndef _POTENTIAL
#define _POTENTIAL


#include "include_libs.h"

class PS_Box;

class PS_Potential {
protected:
    std::string input_command;
    void ramp_check_input(std::istringstream&);
    PS_Box* mybox;
public:
    float energy;
    bool ramp = 0;
    bool allocated = false;

    float *ur, *d_ur;   // [M] potential energy func defined on grid
    float *fA, *d_fA;   // [M*Dim] field of forces acting on A
    float *fB, *d_fB;   // [M*Dim] field of forces acting on B

    std::complex<float> *uk;    // [M] Fourier trans of ur
    cuComplex *d_uk;

    std::complex<float> *fk;    // [M*Dim] Fourier trans of grad u(r)
    cuComplex *d_fk;


    std::complex<float> *virk;       // [M*n_Pcomps] stores k-space virial kernels
    cuComplex *d_virk;

    PS_Potential();
    PS_Potential(std::istringstream&, PS_Box*);
    virtual ~PS_Potential();

    void initializePotential();
    void CalcForces();
    float CalcEnergy();
};

#endif