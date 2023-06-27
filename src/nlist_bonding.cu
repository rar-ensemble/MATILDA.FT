// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "nlist_bonding.h"
#include "nlist.h"
#include <curand_kernel.h>
#include <curand.h>
#include "globals.h"
#include <thrust/copy.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <cmath>
#include <random>
#include <stdio.h>

using namespace std;

NListBonding::~NListBonding() { return; }
NListBonding::NListBonding(istringstream &iss) : NList(iss)

{   

        readRequiredParameter(iss, ad_file);

        std::string line;
        ifstream in(ad_file);
        getline(in, line);
        istringstream str(line);
        
        int ad = 0;
        int count = 0;
        while (str >> ad){
            AD.push_back(ad);
            if (ad == 1){
                d_DONORS.push_back(count); // stores indices within the group
            }

            else if(ad == 0){
                d_ACCEPTORS.push_back(count);
            }
            ++count;
        }

        n_donors = d_DONORS.size();
        n_acceptors = d_ACCEPTORS.size();

        DGRID = (int)ceil((float)(n_donors) / threads);
        AGRID = (int)ceil((float)(n_acceptors) / threads);

}

void NListBonding::MakeNList()
{   

    thrust::fill(d_MASTER_GRID.begin(),d_MASTER_GRID.end(),-1);
    thrust::fill(d_MASTER_GRID_counter.begin(),d_MASTER_GRID_counter.end(),0);

    thrust::fill(d_RN_ARRAY.begin(),d_RN_ARRAY.end(),-1);
    thrust::fill(d_RN_ARRAY_COUNTER.begin(),d_RN_ARRAY_COUNTER.end(),0);

    thrust::fill(d_LOW_DENS_FLAG.begin(), d_LOW_DENS_FLAG.end(), 0);

    d_nlist_bonding_update_grid<<<AGRID, group->BLOCK>>>(d_x, d_Lh, d_L,
        d_MASTER_GRID_counter.data(), d_MASTER_GRID.data(),
        d_Nxx.data(), d_Lg.data(),
        d_LOW_DENS_FLAG.data(),
        d_ACCEPTORS.data(),
        nncells, n_acceptors, ad_hoc_density,
        group->d_index.data(), group->nsites, Dim);

    // Updates n-list for the donors

    d_nlist_bonding_update_nlist<<<DGRID, group->BLOCK>>>(d_x, d_Lh, d_L,
        d_MASTER_GRID_counter.data(), d_MASTER_GRID.data(),
        d_Nxx.data(), d_Lg.data(),
        d_RN_ARRAY.data(), d_RN_ARRAY_COUNTER.data(),
        d_DONORS.data(),
        nncells, n_donors, r_skin, ad_hoc_density,
        group->d_index.data(), group->nsites, Dim);


    // Updates the distribution of acceptors on the grid

    int sum = thrust::reduce(d_LOW_DENS_FLAG.begin(), d_LOW_DENS_FLAG.end(), 0, thrust::plus<int>());
    LOW_DENS_FLAG = float(sum)/float(d_MASTER_GRID_counter.size());
    if (LOW_DENS_FLAG > 0){
        
        cout << "Input density was: " << ad_hoc_density <<" but at least "<< ad_hoc_density + LOW_DENS_FLAG <<" is required"<<endl;
        ad_hoc_density += ceil(LOW_DENS_FLAG*1.5);
        cout << "Increasing the density to " <<  ad_hoc_density <<  " at step " << step << endl;

        d_MASTER_GRID.resize(xyz * ad_hoc_density);                 
        d_RN_ARRAY.resize(group->nsites * ad_hoc_density * nncells);

        // MASTER_GRID.resize(xyz * ad_hoc_density);                 
        // RN_ARRAY.resize(group->nsites * ad_hoc_density * nncells);

        LOW_DENS_FLAG = 0;

        d_nlist_bonding_update_grid<<<AGRID, group->BLOCK>>>(d_x, d_Lh, d_L,
            d_MASTER_GRID_counter.data(), d_MASTER_GRID.data(),
            d_Nxx.data(), d_Lg.data(),
            d_LOW_DENS_FLAG.data(),
            d_ACCEPTORS.data(),
            nncells, n_acceptors, ad_hoc_density,
            group->d_index.data(), group->nsites, Dim);

        // Updates n-list for the donors

        d_nlist_bonding_update_nlist<<<DGRID, group->BLOCK>>>(d_x, d_Lh, d_L,
            d_MASTER_GRID_counter.data(), d_MASTER_GRID.data(),
            d_Nxx.data(), d_Lg.data(),
            d_RN_ARRAY.data(), d_RN_ARRAY_COUNTER.data(),
            d_DONORS.data(),
            nncells, n_donors, r_skin, ad_hoc_density,
            group->d_index.data(), group->nsites, Dim);

    }
}

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
    const int D)
{

    int acceptor_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (acceptor_ind >= n_acceptors)
        return;

    int list_ind = d_ACCEPTORS[acceptor_ind];
    int ind = d_index[list_ind];

    int xi = floor(x[ind * D] / d_Lg[0]);
    int yi = floor(x[ind * D + 1] / d_Lg[1]);
    int zi = floor(x[ind * D + 2] / d_Lg[2]);

    int cell_id;

    int dxx = d_Nxx[0];
    int dyy = d_Nxx[1];
    int dzz = d_Nxx[2];

    if (D == 3){cell_id = xi * dyy * dzz + yi * dzz + zi;}
    else if(D == 2){cell_id = xi * dyy + yi;}

    int insrt_pos = atomicAdd(&d_MASTER_GRID_counter.get()[cell_id], 1);
    if (insrt_pos < ad_hoc_density){
        d_MASTER_GRID[cell_id * ad_hoc_density + insrt_pos] = list_ind;
    }
    else{
        ++d_LOW_DENS_FLAG[list_ind];
    }
    __syncthreads();
}


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
    const int D)
{

    int donor_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (donor_ind >= n_donors)
        return;

    int list_ind = d_DONORS[donor_ind];
    int ind = d_index[list_ind];

    int xi = floor(x[ind * D] / d_Lg[0]);
    int yi = floor(x[ind * D + 1] / d_Lg[1]);
    int zi = floor(x[ind * D + 2] / d_Lg[2]);

    int dxx = d_Nxx[0];
    int dyy = d_Nxx[1];
    int dzz = d_Nxx[2];

    int *ngs = new int[nncells];

    int nxi, nyi, nzi, nid, counter, lnid;
    counter = 0;

    if (D == 3){
        for (int i1 = -1; i1 < 2; i1++)
        {
            for (int i2 = -1; i2 < 2; i2++)
            {
                for (int i3 = -1; i3 < 2; i3++)
                {
                    nxi = (xi + i1 + dxx) % dxx;
                    nyi = (yi + i2 + dyy) % dyy;
                    nzi = (zi + i3 + dzz) % dzz;
                    nid = nxi * dyy * dzz + nyi * dzz + nzi;
                    ngs[counter++] = nid;
                }
            }
        }
    }

    else if (D == 2){
        for (int i1 = -1; i1 < 2; i1++)
        {
            for (int i2 = -1; i2 < 2; i2++)
            {
                nxi = (xi + i1 + dxx) % dxx;
                nyi = (yi + i2 + dyy) % dyy;
                nid = nxi * dyy + nyi;
                ngs[counter] = nid;
                ++counter;
                }
            }
        }

    float my_x[3], dr_arr[3];

    for (int j = 0; j < D; j++){
        my_x[j] = x[ind * D + j];
    }

    for (int i = 0; i < nncells; ++i){
        for (int j = 0; j < d_MASTER_GRID_counter[ngs[i]]; j++){

            float dist = 0.0;                
            float dr_2 = 0.0;
            float dr0;

            lnid = d_MASTER_GRID[ngs[i] * ad_hoc_density + j];
    
            nid = d_index[lnid];

            for (int j = 0; j < D; j++){
                dr0 = my_x[j] - x[nid * D + j];

                if (dr0 >  Lh[j]){dr_arr[j] = -1.0 * (L[j] - dr0);}
                else if (dr0 < -1.0 * Lh[j]){dr_arr[j] = (L[j] + dr0);}
                else{dr_arr[j] = dr0;}

                dr_2 += dr_arr[j] * dr_arr[j];
            }
            if (dr_2 > 1.0E-5f) {
                dist = sqrt(dr_2);
            }
            else{
                dist = 0.0f;
            }
            if (dist <= r_skin){
                int insrt_pos = atomicAdd(&d_RN_ARRAY_COUNTER.get()[list_ind], 1);
                if (insrt_pos < ad_hoc_density * nncells){
                    d_RN_ARRAY[list_ind * ad_hoc_density*nncells + insrt_pos] = lnid;
                }
            }
        }
    } 
    delete[] ngs;
    __syncthreads();       
}