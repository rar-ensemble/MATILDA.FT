// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "Extraforce.h"
#include "nlist.h"
#include "nlist_bonding.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <ctime>
#include <thrust/copy.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <cmath>
#include <random>
#include <stdio.h>
#include <ctime>

__global__ void d_break_bonds_lewis_serial(
    const float *x,
    thrust::device_ptr<int> d_BONDS,
    thrust::device_ptr<int> d_RN_ARRAY,
    thrust::device_ptr<int> d_RN_ARRAY_COUNTER,
    thrust::device_ptr<int> d_BONDED,
    int n_bonded,
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
    float* d_charges,
    int grid_per_partic,
    float* d_electrostatic_potential,
    int* d_grid_inds,
    float* d_grid_W,
    int rndid);


__global__ void d_make_bonds_lewis_serial(
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
    float r_n,
    float qind,
    float *L,
    float *Lh,
    int D,
    float* d_charges,
    int grid_per_partic,
    float* d_electrostatic_potential,
    int* d_grid_inds,
    float* d_grid_W,
    int rndid);


#ifndef _EXTRAFORCE_LEWIS_SERIAL
#define _EXTRAFORCE_LEWIS_SERIAL


class LewisSerial : public ExtraForce {
protected:
    
    int bond_freq, n_free, n_bonded;  
    int GRID;

    float k_spring, e_bond, r0, qind;   
    
    std::string file_name;

    thrust::host_vector<int> AD, BONDS, BONDED, FREE;
    thrust::device_vector<int> d_BONDS, d_FREE, d_BONDED;//, d_FLAG_LIST, d_AD;
    
    thrust::device_vector<float> d_VirArr;
    thrust::host_vector<float> VirArr;

    NListBonding* nlist;
    int offset;

    //Ramp parameters
    std::string ramp_string;
    int RAMP_FLAG, ramp_interval, ramp_t_end, ramp_reps, ramp_counter;
    float e_bond_final, delta_e_bond;

public:
	~LewisSerial();
	LewisSerial(std::istringstream&);
    void AddExtraForce() override;
    void WriteBonds();
    // void UpdateHD(void);
    void UpdateVirial(void) override;

};

#endif

