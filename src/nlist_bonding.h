// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "nlist.h"

__global__ void d_nlist_bonding_update_grid(
    const float *x,
    const float *Lh,
    const float *L,
    thrust::device_ptr<int> d_MASTER_GRID_counter,
    thrust::device_ptr<int> d_MASTER_GRID,
    thrust::device_ptr<int> d_Nxx,
    thrust::device_ptr<float> d_Lg,
    thrust::device_ptr<int> d_LOW_DENS_FLAG,
    thrust::device_ptr<int> d_ACCEPTORS,
    const int nncells,
    const int n_acceptors,
    const int ad_hoc_density,
    thrust::device_ptr<int> d_index, 
    const int ns,      
    const int D);

__global__ void d_nlist_bonding_update_nlist(
    const float *x,
    const float *Lh,
    const float *L,
    thrust::device_ptr<int> d_MASTER_GRID_counter,
    thrust::device_ptr<int> d_MASTER_GRID,
    thrust::device_ptr<int> d_Nxx,
    thrust::device_ptr<float> d_Lg,
    thrust::device_ptr<int> d_RN_ARRAY,
    thrust::device_ptr<int> d_RN_ARRAY_COUNTER,
    thrust::device_ptr<int> d_DONORS,
    const int nncells,
    const int n_donors,
    const float r_skin,
    const int ad_hoc_density,
    thrust::device_ptr<int> d_index, 
    const int ns,      
    const int D);   



#ifndef _NLIST_BONDING
#define _NLIST_BONDING


class NListBonding : public NList {
protected:
                      
public:
	~NListBonding();
	NListBonding(std::istringstream&);


    int n_donors, n_acceptors;
    int DGRID, AGRID;

    std::string ad_file;

    thrust::host_vector<int> AD;
    thrust::device_vector<int> d_DONORS, d_ACCEPTORS;

    // Override functions

    void MakeNList() override;


    // Additional functions

    int getDonors();
    int getAcceptors();
    
};

#endif

