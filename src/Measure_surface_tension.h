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


#ifndef _MEASURE_SURFACE_TENSION
#define _MEASURE_SURFACE_TENSION


class SurfaceTension : public Measure {
protected:

    int freq;
public:
	~SurfaceTension();
	SurfaceTension(std::istringstream&);

    int LogCheck();

    void RestoreState() override;
    void PerturbState() override;
    void WriteLog();

    std::string file_name;

	float p_dx[3];
    float p_L[3];
	float p_gvol;
    float delta;
    float scales[3];
    float *p_x;
    float *p_x0;

    float gvol0;

};

#endif



