// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#ifndef _FIELD_COMP
#define _FIELD_COMP

class FieldComponent {
public:
    float* rho, ** force, * d_rho, * d_force;

    //__global__ void ZeroDeviceGradient(int);
    void ZeroGradient();
    FieldComponent();
    ~FieldComponent();
};

#endif  