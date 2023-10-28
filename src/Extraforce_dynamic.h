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



__global__ void d_make_bonds(
    const float *x,
    float* f,    
    thrust::device_ptr<int> d_BONDS,
    thrust::device_ptr<int> d_ACCEPTORS,
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
   thrust::device_ptr<int> d_mbbond,
    float *L,
    float *Lh,
    int D);

__global__ void d_break_bonds(
    const float *x,
    thrust::device_ptr<int> d_BONDS,
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
   thrust::device_ptr<int> d_mbbond,
    float *L,
    float *Lh,
    int D);

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

#ifndef _EXTRAFORCE_DYNAMIC
#define _EXTRAFORCE_DYNAMIC


class Dynamic : public ExtraForce {
protected:
    
    int bond_freq, n_free_donors, n_free_acceptors, n_bonded;  
    int GRID;
    thrust::device_vector<int> d_mbbond;
    thrust::host_vector<int> mbbond;

    float k_spring, e_bond, r0, qind;   
    
    std::string file_name;

    thrust::host_vector<int> AD, BONDS, BONDED, FREE;
    thrust::device_vector<int> d_BONDS, d_FREE, d_BONDED;//, d_FLAG_LIST, d_AD;
    
    thrust::device_vector<float> d_VirArr;
    thrust::host_vector<float> VirArr;

    // NListBonding* nlist;
    int offset;
    float r_n;
    //Ramp parameters
    std::string ramp_string;
    int RAMP_FLAG, ramp_interval, ramp_t_end, ramp_reps, ramp_counter;
    float e_bond_final, delta_e_bond;

    int n_donors, n_acceptors, acceptor_tag, donor_tag;

    std::string ad_file;

    thrust::host_vector<int> DONORS, ACCEPTORS, FREE_ACCEPTORS, S_ACCEPTORS;
    thrust::device_vector<int> d_DONORS, d_ACCEPTORS, d_S_ACCEPTORS, d_FREE_ACCEPTORS;

    thrust::device_vector<int> d_RN_ARRAY;
    thrust::device_vector<int> d_RN_ARRAY_COUNTER;

    thrust::host_vector<int> RN_ARRAY;
    thrust::host_vector<int> RN_ARRAY_COUNTER;
    thrust::default_random_engine g;

public:
	~Dynamic();
	Dynamic(std::istringstream&);
    void AddExtraForce() override;
    void WriteBonds();
    // void UpdateHD(void);
    void UpdateVirial(void) override;

};

#endif
