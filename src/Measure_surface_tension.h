// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "Measure.h"

// __global__ void d_ExtraForce_midpush(
//     float* f,               // [ns*Dim], particle forces
//     const float* x,         // [ns*Dim], particle positions
//     const float fmag,       // Force magnitude
//     thrust::device_ptr<int> d_index,   // List of sites in the group
//     const float* Lh,        // [Dim] Half box length
//     const int ns,
//     const int axis,           // Number of sites in the list
//     const int D);


#ifndef _EXTRAFORCE_SURFACE_TENSION
#define _EXTRAFORCE_SURFACE_TENSION


class SurfaceTension : public Measure {
protected:
    float delta;         // Force magnitude to push particles
    int freq;
public:
	~SurfaceTension();
	SurfaceTension(std::istringstream&);
    void AddMeasure() override;
    int LogCheck();

	float st_dx[3];
    float st_L[3];
	float st_gvol;
    float delta;
    float scales[3];
    float *st_x;

};

#endif



