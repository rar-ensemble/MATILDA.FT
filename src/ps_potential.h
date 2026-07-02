// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#ifndef _POTENTIAL
#define _POTENTIAL


#include "include_libs.h"

class PS_Box;

class PS_Potential {
protected:
    std::string input_command;
    float Ao_initial, Ao_final;     // initial and final prefactor values for potential
    PS_Box* mybox;
public:
    float energy;
    bool ramp = 0;
    bool allocated = false;
    int ramp_check_input(std::istringstream&, float);

    float *ur, *d_ur;   // [M] potential energy func defined on grid
    float *fI, *d_fI;   // [M*Dim] field of forces acting on I
    float *fJ, *d_fJ;   // [M*Dim] field of forces acting on J

    std::complex<float> *uk;    // [M] Fourier trans of ur
    cuComplex *d_uk;

    std::complex<float> *fk;    // [M*Dim] Fourier trans of grad u(r)
    cuComplex *d_fk;

    std::complex<float> *virk;       // [M*n_Pcomps] stores k-space virial kernels
    cuComplex *d_virk;

    std::string grpI, grpJ; // Groups on which this potential acts
    int Iind, Jind;     // Group indices on which this potential acts

    PS_Potential();
    PS_Potential(std::istringstream&, PS_Box*);
    virtual ~PS_Potential();

    virtual void initializePotential();
    
    // These are virtual so they can be overriden for non-2 body potentials
    virtual void CalcForces();      
    virtual float CalcEnergy();
    virtual void update_prefactor(const int, const int);

    virtual void initBinaryOutput();    
    virtual void writeBinaryOutput();   
};

#endif