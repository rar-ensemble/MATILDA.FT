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


/*
Make bonds
*/


// __global__ void d_make_bonds_lewis_full_2(
//     const float *x,
//     float* f,    
//     thrust::host_vector<int>& d_BONDS,
//     thrust::host_vector<int>& d_RN_ARRAY,
//     thrust::host_vector<int>& d_RN_ARRAY_COUNTER,
//     thrust::host_vector<int>& d_FREE,
//     thrust::host_vector<float>& d_VirArr,
//     int random_id,
//     int nncells,
//     int ad_hoc_density,
//     thrust::host_vector<int>& d_index, 
//     const int ns,        
//     curandState *d_states,
//     float k_spring,
//     float e_bond,
//     float r0,
//     float r_n,
//     float qind,
//     float *L,
//     float *Lh,
//     int D,
//     float* d_charges,
//     thrust::host_vector<int>& d_lewis_vect,
//     thrust::host_vector<float>& d_dU_lewis);   

//     /* Break Bonds*/

// __global__ void d_break_bonds_lewis_full_1(
//     const float *x,
//     float* f,
//     thrust::host_vector<int>& d_BONDS,
//     thrust::host_vector<int>& d_RN_ARRAY,
//     thrust::host_vector<int>& d_RN_ARRAY_COUNTER,
//     thrust::host_vector<int>& d_BONDED,
//     thrust::host_vector<float>& d_VirArr,
//     int random_id,
//     int nncells,
//     int ad_hoc_density,
//     thrust::host_vector<int>& d_index, 
//     const int ns,        
//     curandState *d_states,
//     float k_spring,
//     float e_bond,
//     float r0,
//     float r_n,
//     float qind,
//     float *L,
//     float *Lh,
//     int D,
//     float* d_charges,
//     thrust::host_vector<int>& d_lewis_vect,
//     thrust::host_vector<float>& d_dU_lewis);   


// __global__ void d_break_bonds_lewis_full_2(
//     const float *x,
//     float* f,
//     thrust::host_vector<int>& d_BONDS,
//     thrust::host_vector<int>& d_RN_ARRAY,
//     thrust::host_vector<int>& d_RN_ARRAY_COUNTER,
//     thrust::host_vector<int>& d_BONDED,
//     thrust::host_vector<float>& d_VirArr,
//     int random_id,
//     int nncells,
//     int ad_hoc_density,
//     thrust::host_vector<int>& d_index, 
//     const int ns,        
//     curandState *d_states,
//     float k_spring,
//     float e_bond,
//     float r0,
//     float r_n,
//     float qind,
//     float *L,
//     float *Lh,
//     int D,
//     float* d_charges,
//     thrust::host_vector<int>& d_lewis_vect,
//     thrust::host_vector<float>& d_dU_lewis);   




#ifndef _EXTRAFORCE_LEWIS_CPU
#define _EXTRAFORCE_LEWIS_CPU


class LewisCPU : public ExtraForce {
protected:
    
    int bond_freq, n_free, n_bonded;  
    int GRID;
    int random_ind;

    float k_spring, e_bond, r0, qind;   
    
    std::string file_name;

    thrust::host_vector<int> AD, BONDS, BONDED, FREE, lewis_vect;
    thrust::device_vector<int> d_BONDS, d_FREE, d_BONDED, d_lewis_vect;//, d_FLAG_LIST, d_AD;
    
    thrust::device_vector<float> d_VirArr, d_dU_lewis;
    thrust::host_vector<float> VirArr, dU_lewis;

    NListBonding* nlist;
    int offset;

    //Ramp parameters
    std::string ramp_string;
    int RAMP_FLAG, ramp_interval, ramp_t_end, ramp_reps, ramp_counter;
    float e_bond_final, delta_e_bond;

    float dU_lewis;
    int ind1, ind2, lewis_proceed;

public:


	~LewisCPU();
	LewisCPU(std::istringstream&);
    void AddExtraForce() override;
    void WriteBonds();
    // void UpdateHD(void);
    void UpdateVirial(void) override;

void LewisCPU::make_bonds_cpu(
    const float *x,
    const float* f,    
    float *L,
    float *Lh,
    int D,
    float* charges);


    void make_break_cpu();



};

#endif

