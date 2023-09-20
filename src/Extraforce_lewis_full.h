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

__global__ void d_make_bonds_lewis_full_1(
    const float *x,
    float* f,    
    thrust::device_ptr<int> d_BONDS,
    thrust::device_ptr<int> d_RN_ARRAY,
    thrust::device_ptr<int> d_RN_ARRAY_COUNTER,
    thrust::device_ptr<int> d_FREE,
    thrust::device_ptr<float> d_VirArr,
    int random_id,
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
    thrust::device_ptr<int> d_lewis_vect,
    thrust::device_ptr<float> d_dU_lewis);   



__global__ void d_make_bonds_lewis_full_2(
    const float *x,
    float* f,    
    thrust::device_ptr<int> d_BONDS,
    thrust::device_ptr<int> d_RN_ARRAY,
    thrust::device_ptr<int> d_RN_ARRAY_COUNTER,
    thrust::device_ptr<int> d_FREE,
    thrust::device_ptr<float> d_VirArr,
    int random_id,
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
    thrust::device_ptr<int> d_lewis_vect,
    thrust::device_ptr<float> d_dU_lewis);   

    /* Break Bonds*/

__global__ void d_break_bonds_lewis_full_1(
    const float *x,
    float* f,
    thrust::device_ptr<int> d_BONDS,
    thrust::device_ptr<int> d_RN_ARRAY,
    thrust::device_ptr<int> d_RN_ARRAY_COUNTER,
    thrust::device_ptr<int> d_BONDED,
    thrust::device_ptr<float> d_VirArr,
    int random_id,
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
    int ind1,
    int ind2,
    float dU_lewis);   


__global__ void d_break_bonds_lewis_full_2(
    const float *x,
    float* f,
    thrust::device_ptr<int> d_BONDS,
    thrust::device_ptr<int> d_RN_ARRAY,
    thrust::device_ptr<int> d_RN_ARRAY_COUNTER,
    thrust::device_ptr<int> d_BONDED,
    thrust::device_ptr<float> d_VirArr,
    int random_id,
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
    int ind1,
    int ind2,
    float dU_lewis);   




#ifndef _EXTRAFORCE_LEWIS_FULL
#define _EXTRAFORCE_LEWIS_FULL


class LewisFull : public ExtraForce {
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

public:
	~LewisFull();
	LewisFull(std::istringstream&);
    void AddExtraForce() override;
    void WriteBonds();
    // void UpdateHD(void);
    void UpdateVirial(void) override;

};

#endif

