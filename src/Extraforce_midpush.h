// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "Extraforce.h"

__global__ void d_ExtraForce_midpush(
    float* f,               // [ns*Dim], particle forces
    const float* x,         // [ns*Dim], particle positions
    const float fmag,       // Force magnitude
    thrust::device_ptr<int> d_index,   // List of sites in the group
    const float* Lh,        // [Dim] Half box length
    const int ns,
    const int axis,           // Number of sites in the list
    const int D);


#ifndef _EXTRAFORCE_MIDPUSH
#define _EXTRAFORCE_MIDPUSH


class Midpush : public ExtraForce {
protected:
    float force_magnitude;         // Force magnitude to push particles
    int axis;
public:
	~Midpush();
	Midpush(std::istringstream&);
    void AddExtraForce() override;
    void UpdateVirial(void) override;
};

#endif



