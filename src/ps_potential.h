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

    thrust::host_vector<float> ur;      // [M] potential energy func defined on grid
    thrust::device_vector<float> d_ur;  // [M] potential energy func defined on grid

    thrust::host_vector<float> fA;      // [M*Dim] field of forces acting on A
    thrust::device_vector<float> d_fA;  // [M*Dim] field of forces acting on A

    thrust::host_vector<float> fB;      // [M*Dim] field of forces acting on B
    thrust::device_vector<float> d_fB;  // [M*Dim] field of forces acting on B

    thrust::host_vector<thrust::complex<float>> uk;     // [M] Fourier transform of grad u(r)
    thrust::device_vector<thrust::complex<float>> d_uk; // [M] Fourier transform of grad u(r)

    thrust::host_vector<thrust::complex<float>> fk;     // [M*Dim] Fourier transform of grad u(r)
    thrust::device_vector<thrust::complex<float>> d_fk; // [M*Dim] Fourier transform of grad u(r)

    thrust::host_vector<thrust::complex<float>> virk;       // [M*n_Pcomps] stores k-space virial kernels
    thrust::device_vector<thrust::complex<float>> d_virk;   // [M*n_Pcomps] stores k-space virial kernels
    

    PS_Potential();
    PS_Potential(std::istringstream&, PS_Box*);
    virtual ~PS_Potential();

    void Initialize_Potential();
    void CalcForces();
    float CalcEnergy();
};

#endif