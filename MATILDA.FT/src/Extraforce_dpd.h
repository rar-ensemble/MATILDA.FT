// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "Extraforce.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__global__ void d_ExtraForce_dpd_update_forces(
    const float *x, // [ns*Dim], particle positions
    float *f,
    const float *v,
    const float *Lh,
    const float *L,
    thrust::device_ptr<int> d_MASTER_GRID_counter,
    thrust::device_ptr<int> d_MASTER_GRID,
    thrust::device_ptr<int> d_RN_ARRAY,
    thrust::device_ptr<int> d_RN_ARRAY_COUNTER,
    thrust::device_ptr<int> d_Nxx,
    thrust::device_ptr<float> d_Lg,
    thrust::device_ptr<int> d_index, // List of sites in the group
    const int ns,         // Number of sites in the list
    const int D,
    curandState *d_states,
    const int step,
    const int ad_hoc_density,
    const int nncells,
    const double sigma,
    const double r_cutoff,
    const float delt);


#ifndef _EXTRAFORCE_DPD
#define _EXTRAFORCE_DPD


class DPD : public ExtraForce {
protected:
                      
    int nlist_freq;
    double sigma, r_cutoff;   

    thrust::host_vector<int> BONDS;
    thrust::device_vector<int> d_BONDS;

public:
	~DPD();
	DPD(std::istringstream&);
    void AddExtraForce() override;
    void WriteBonds();
    void UpdateHD(void);
    void UpdateVirial(void) override;
};

#endif

