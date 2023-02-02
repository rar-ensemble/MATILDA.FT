// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "Extraforce.h"
#include "nlist.h"
#include "nlist_bonding.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

__global__ void d_break_bonds(
    const float *x,
    thrust::device_ptr<int> d_BONDS,
    thrust::device_ptr<int> d_RN_ARRAY,
    thrust::device_ptr<int> d_RN_ARRAY_COUNTER,
    thrust::device_ptr<int> d_BONDED,
    int n_bonded,
    thrust::device_ptr<int> d_index, 
    const int ns,        
    curandState *d_states,
    float k_spring,
    float e_bond,
    float qind,
    const int D,
    float* d_charges);

__global__ void d_make_bonds(
    const float *x,
    float* f,    
    thrust::device_ptr<int> d_BONDS,
    thrust::device_ptr<int> d_RN_ARRAY,
    thrust::device_ptr<int> d_RN_ARRAY_COUNTER,
    thrust::device_ptr<int> d_FREE,
    thrust::device_ptr<float> d_VirArr,
    int n_free,
    int nncells,
    int ad_hoc_density,
    thrust::device_ptr<int> d_index, 
    const int ns,        
    curandState *d_states,
    float k_spring,
    float e_bond,
    float r0,
    float qind,
    float *L,
    float *Lh,
    int D,
    float* d_charges);


__global__ void d_update_forces(
    const float* x,    
    float* f,        // [ns*Dim], particle positions
    const float* d_L,
    const float* d_Lh,
    float k_spring,
    float e_bond,
    float r0,
    thrust::device_ptr<int> d_BONDS,
    thrust::device_ptr<int> d_BONDED,
    thrust::device_ptr<float> d_VirArr,
    const int n_bonded,
    thrust::device_ptr<int> d_index,   // List of sites in the group
    const int ns,           // Number of sites in the list
    const int D);   

#ifndef _EXTRAFORCE_LEWIS
#define _EXTRAFORCE_LEWIS


class Lewis : public ExtraForce {
protected:
    
    int bond_freq, n_free, n_bonded;  
    int BGRID, FGRID;

    float k_spring, e_bond, r0, qind;   
    
    std::string file_name;

    thrust::host_vector<int> AD, BONDS;
    thrust::device_vector<int> d_BONDS, d_FREE, d_BONDED;//, d_FLAG_LIST, d_AD;
    
    thrust::device_vector<float> d_VirArr;
    thrust::host_vector<float> VirArr;

    NListBonding* nlist;

public:
	~Lewis();
	Lewis(std::istringstream&);
    void AddExtraForce() override;
    void WriteBonds();
    // void UpdateHD(void);
    void UpdateVirial(void) override;

};

#endif

