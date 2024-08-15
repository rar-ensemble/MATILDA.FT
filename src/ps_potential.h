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
    float *u;
    float **f;
    
    bool ramp = false;
    bool allocated = false;

    PS_Potential();
    PS_Potential(std::istringstream&, PS_Box*);
    virtual ~PS_Potential();

    void Initialize_Potential();
    void CalcForces();
    float CalcEnergy();
};

#endif