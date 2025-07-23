// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "Extraforce_dpd.h"
#include <curand_kernel.h>
#include <curand.h>
#include "globals.h"
#include "timing.h"
#include <thrust/copy.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <cmath>
#include <random>
#include <stdio.h>

#define EPSILON 1.0e-10

using namespace std;

DPD::~DPD() { return; }
void DPD::UpdateVirial(void){return;};

DPD::DPD(istringstream &iss) : ExtraForce(iss)
{

    readRequiredParameter(iss, nlist_name);
	nlist_index = get_nlist_id(nlist_name);
    nlist = NLists.at(nlist_index);
    cout <<"N-list index: " << nlist_index << ", n-list name: " << nlist_name << endl;
    cout << "DPD thermostat active..." << endl;

    readRequiredParameter(iss, sigma); // dissipative force coefficient
    readRequiredParameter(iss, r_cutoff); // cutoff radius

    cout << "sigma (random force): " << sigma << endl; 
    cout << "gamma (diussipative force): " << sigma * sigma/2.0 << endl; 
    cout << "r_cutoff: " << r_cutoff << endl;

    cout << "Group size: " << this->group->nsites << endl;
}

void DPD::AddExtraForce()
{   

    if (nlist->CheckTrigger()){
        int tmp_time = time(0);
        nlist->MakeNList();
        nl_time += time(0) - tmp_time;
    }

    int tmp_time = time(0);

    d_ExtraForce_dpd_update_forces<<<group->GRID, group->BLOCK>>>(d_x, d_f, d_v, d_Lh, d_L,
        nlist->d_MASTER_GRID_counter.data(), nlist->d_MASTER_GRID.data(),
        nlist->d_RN_ARRAY.data(),nlist->d_RN_ARRAY_COUNTER.data(),
        nlist->d_Nxx.data(), nlist->d_Lg.data(),
        group->d_index.data(), group->nsites, Dim, d_states, step,
        nlist->ad_hoc_density, nlist->nncells,
        sigma, r_cutoff, delt);
        
    DPD_time += time(0) - tmp_time;
}

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
    const float delt){

    int list_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (list_ind >= ns)
        return;

    int ind = d_index[list_ind];

    float my_x[3], my_v[3], my_f[3];
    
    for (int j = 0; j < D; j++){
        my_x[j] = x[ind * D + j];
        my_v[j] = v[ind * D + j];
        my_f[j] = 0.f;
    }

    for (int i = 0; i < d_RN_ARRAY_COUNTER[list_ind]; i++){

        int lnid = d_RN_ARRAY[list_ind * ad_hoc_density * nncells + i];

        double v_dot = 0.0, dr_sq = 0.0, dr0 = 0.0;
        double dr_arr[3], dv_arr[3], e_ij[3];
        double gamma = sigma*sigma/2.0;


        int nid = d_index[lnid]; // neighbor ID

        for (int j = 0; j < D; j++)
        {
            dr0 = my_x[j] - x[nid * D + j];
            dv_arr[j] = my_v[j] - v[nid * D + j];

            if (dr0 >  Lh[j]){dr_arr[j] = -1.0 * (L[j] - dr0);} // pbc
            else if (dr0 < -1.0 * Lh[j]){dr_arr[j] = (L[j] + dr0);}
            else{dr_arr[j] = dr0;}

            dr_sq += dr_arr[j] * dr_arr[j];
        }

        //if (dr_sq > 1.0E-5f) 
        double dr = sqrt(dr_sq); // scalar distance between particles
        if (dr > EPSILON){
            double drinv = 1.0/dr;
            double invsqts = 1/sqrt(delt);

            for (int j = 0; j < D; ++j){
                e_ij[j] = dr_arr[j] * drinv; // unit vector in j = x, y ,z
                v_dot += dv_arr[j] * dr_arr[j];//e_ij[j];
            }

            double w_r = 0.0f;
            double w_d = 0.0f;

            w_r = (1 - dr/r_cutoff);
            w_d = w_r * w_r;

            if (dr < r_cutoff){
                for (int j = 0; j < D; ++j)
                {   
                    curandState l_state;
                    l_state = d_states[ind];
                    double theta = curand_normal(&l_state);
                    d_states[ind] = l_state;
                    double dpd_force = sigma * w_r * theta * invsqts * e_ij[j] - gamma * w_d * v_dot * drinv * e_ij[j];

                    if (!isnan(dpd_force)){
                        my_f[j] += dpd_force;
                        atomicAdd(&f[nid*D + j], -1.0*dpd_force);
                    }
                }//j=0:D
            }// dr < r_cutoff
        } //if dr < EPSILON
    }// for i=0:d_RN_ARRAY_COUNTER

    // Accumulate force on 'ind'
    for ( int j=0 ; j<D ; j++ ) {
        atomicAdd(&f[ind*D + j], my_f[j]);
    }
}

