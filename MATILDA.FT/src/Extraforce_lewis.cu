// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "Extraforce_lewis.h"
#include <curand_kernel.h>
#include <curand.h>
#include "globals.h"
#include <thrust/copy.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <cmath>
#include <random>
#include <stdio.h>
#include <ctime>


#define EPSILON 1.0e-10

using namespace std;


Lewis::~Lewis() { return; }

Lewis::Lewis(istringstream &iss) : ExtraForce(iss)
{

    DynamicBonds.push_back(this);

    std::srand(std::time(0));

    readRequiredParameter(iss, nlist_name);
	nlist_index = get_nlist_id(nlist_name);
    nlist = dynamic_cast<NListBonding*>(NLists.at(nlist_index));

    cout << "Lewis bonds active!" << endl;
    cout <<"N-list index: " << nlist_index << ", n-list name: " << nlist_name << endl;


    readRequiredParameter(iss, k_spring);
    cout << "k_spring: " << k_spring << endl;
    readRequiredParameter(iss, e_bond);
    cout << "e_bond: " << e_bond << endl;
    readRequiredParameter(iss, r0);
    cout << "r0: " << r0 << endl;
    readRequiredParameter(iss, qind);
     cout << "qind: " << qind << endl;
    readRequiredParameter(iss, bond_freq);
    cout << "bond_freq: " << bond_freq << endl;
    readRequiredParameter(iss, bond_log_freq);
    cout << "bond_log_freq: " << bond_freq << endl;
    readRequiredParameter(iss, file_name);
    cout << "output_file: " << file_name << endl;

    cout << "Group size: " << group->nsites << endl;
    cout << "Donors: " << nlist->n_donors << endl;
    cout << "Acceptors: " << nlist->n_acceptors << endl;



    d_BONDS.resize(2 * group->nsites); 
    BONDS.resize(2 * group->nsites);     

    // BONDS - stores bonding information
    // group_id [n_bonds] [bonded_parter_group_id]

    for (int j = 0; j < group->nsites; j++)
    {
        BONDS[2 * j] = 0;
        BONDS[2 * j + 1] = -1;
    }

    d_BONDS = BONDS;

    d_VirArr.resize(5 * group->nsites); 
    for (int i = 0; i < 5 * group->nsites; ++i){
        d_VirArr[i] = 0.0f;
    }

    d_BONDED.resize(0);
    d_FREE.resize(0);


    for (int i = 0; i < nlist->n_donors; ++i){
        d_FREE.push_back(nlist->d_DONORS[i]);
    }

    n_bonded = 0;
    n_free = d_FREE.size();

    BGRID = (int)ceil((float)(n_bonded) / threads);
    FGRID = (int)ceil((float)(n_free) / threads);
    if (BGRID == 0) BGRID = 1;
    if (FGRID == 0) FGRID = 1;
}


void Lewis::AddExtraForce()
{   

    if (step % bond_freq == 0 && step >= bond_freq){
        if (nlist->CheckTrigger() == 1){
            nlist->MakeNList();
            }
        
        int rnd = random()%2; //decide move sequence

        if (rnd == 0){
            if (n_free > 0){
                d_make_bonds<<<FGRID, group->BLOCK>>>(d_x,d_f,
                    d_BONDS.data(),
                    nlist->d_RN_ARRAY.data(), nlist->d_RN_ARRAY_COUNTER.data(),
                    d_FREE.data(), d_VirArr.data(), n_free,
                    nlist->nncells, nlist->ad_hoc_density,
                    group->d_index.data(), group->nsites, d_states,
                    k_spring, e_bond, r0, nlist->r_n, qind, d_L, d_Lh, Dim, d_charges);

                d_BONDED.resize(0);
                d_FREE.resize(0);


                for (int i = 0; i < group->nsites; ++i){
                    if (nlist->AD[i] == 1 && d_BONDS[2*i] == 1){
                        d_BONDED.push_back(i);
                    }
                    else if (nlist->AD[i] == 1 && d_BONDS[2*i] == 0){
                        d_FREE.push_back(i);
                    }
                }

                n_bonded = d_BONDED.size();
                n_free = d_FREE.size();

                BGRID = (int)ceil((float)(n_bonded) / threads);
                FGRID = (int)ceil((float)(n_free) / threads);
                if (BGRID == 0) BGRID = 1;
                if (FGRID == 0) FGRID = 1;

                //Update charges
                if (qind != 0.0){
                    cudaMemcpy(charges, d_charges, ns * sizeof(float), cudaMemcpyDeviceToHost);
                }
            } 
            if (n_bonded > 0){
                d_break_bonds<<<FGRID, group->BLOCK>>>(d_x,
                    d_BONDS.data(),
                    nlist->d_RN_ARRAY.data(), nlist->d_RN_ARRAY_COUNTER.data(),
                    d_BONDED.data(),n_bonded,
                    nlist->nncells, nlist->ad_hoc_density,
                    group->d_index.data(), group->nsites, d_states,
                    k_spring, e_bond, r0, qind, d_L, d_Lh, Dim, d_charges);
            } 
        }

        else {
            if (n_bonded > 0){
                d_break_bonds<<<FGRID, group->BLOCK>>>(d_x,
                    d_BONDS.data(),
                    nlist->d_RN_ARRAY.data(), nlist->d_RN_ARRAY_COUNTER.data(),
                    d_BONDED.data(),n_bonded,
                    nlist->nncells, nlist->ad_hoc_density,
                    group->d_index.data(), group->nsites, d_states,
                    k_spring, e_bond, r0, qind, d_L, d_Lh, Dim, d_charges);

                d_BONDED.resize(0);
                d_FREE.resize(0);


                for (int i = 0; i < group->nsites; ++i){
                    if (nlist->AD[i] == 1 && d_BONDS[2*i] == 1){
                        d_BONDED.push_back(i);
                    }
                    else if (nlist->AD[i] == 1 && d_BONDS[2*i] == 0){
                        d_FREE.push_back(i);
                    }
                }

                n_bonded = d_BONDED.size();
                n_free = d_FREE.size();

                BGRID = (int)ceil((float)(n_bonded) / threads);
                FGRID = (int)ceil((float)(n_free) / threads);
                if (BGRID == 0) BGRID = 1;
                if (FGRID == 0) FGRID = 1;

                //Update charges
                if (qind != 0.0){
                    cudaMemcpy(charges, d_charges, ns * sizeof(float), cudaMemcpyDeviceToHost);
                }
            } 
            if (n_free > 0){
                d_make_bonds<<<FGRID, group->BLOCK>>>(d_x,d_f,
                    d_BONDS.data(),
                    nlist->d_RN_ARRAY.data(), nlist->d_RN_ARRAY_COUNTER.data(),
                    d_FREE.data(), d_VirArr.data(), n_free,
                    nlist->nncells, nlist->ad_hoc_density,
                    group->d_index.data(), group->nsites, d_states,
                    k_spring, e_bond, r0, nlist->r_n, qind, d_L, d_Lh, Dim, d_charges);

                }
        }


        // update the bonded array

        d_BONDED.resize(0);
        d_FREE.resize(0);


        for (int i = 0; i < group->nsites; ++i){
            if (nlist->AD[i] == 1 && d_BONDS[2*i] == 1){
                d_BONDED.push_back(i);
            }
            else if (nlist->AD[i] == 1 && d_BONDS[2*i] == 0){
                d_FREE.push_back(i);
            }
        }

        n_bonded = d_BONDED.size();
        n_free = d_FREE.size();

        BGRID = (int)ceil((float)(n_bonded) / threads);
        FGRID = (int)ceil((float)(n_free) / threads);
        if (BGRID == 0) BGRID = 1;
        if (FGRID == 0) FGRID = 1;

        //Update charges
        if (qind != 0.0){
            cudaMemcpy(charges, d_charges, ns * sizeof(float), cudaMemcpyDeviceToHost);
        }

    } // end if (step % lewis_bond_freq == 0 && step >= bond_freq)

    if(step >= bond_freq){
        d_update_forces<<<BGRID, group->BLOCK>>>(d_x, d_f, d_L, d_Lh,
            k_spring, e_bond, r0,
            d_BONDS.data(), d_BONDED.data(), d_VirArr.data(), n_bonded,
            group->d_index.data(), group->nsites, Dim);
    }

    if (step == 0){
        const char* fname = file_name.c_str();
        remove(fname);
        remove("bonding_info");
    }

    if (step % bond_log_freq == 0 && step >= bond_freq)
    {
        Lewis::WriteBonds();
    }
}


/*
Updates forces acting on particles due to dynamic bonds
*/

__global__ void d_update_forces(
    const float *x,
    float *f,
    const float *L,
    const float *Lh,
    float k_spring,
    float e_bond,
    float r0,
    thrust::device_ptr<int> d_BONDS,
    thrust::device_ptr<int> d_BONDED,
    thrust::device_ptr<float> d_VirArr,
    int n_bonded,
    thrust::device_ptr<int> d_index, // List of sites in the group
    const int ns,         // Number of sites in the list
    const int D)
{

    int tmp_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (tmp_ind >= n_bonded)
        return;

    int list_ind = d_BONDED[tmp_ind];
    int ind = d_index[list_ind];
    int lnid = d_BONDS[2 * list_ind + 1];
    int nid = d_index[lnid];

    double dr_sq = 0.0;
    double dr0;
    double dr_arr[3];
    double delr;
    double mf;
    double dU;

    for (int j = 0; j < D; j++){

        dr0 = x[ind * D + j] - x[nid * D + j];

        if (dr0 >  Lh[j]){dr_arr[j] = -1.0 * (L[j] - dr0);}
        else if (dr0 < -1.0 * Lh[j]){dr_arr[j] = (L[j] + dr0);}
        else{dr_arr[j] = dr0;}

        dr_sq += dr_arr[j] * dr_arr[j];
    }

    double mdr = sqrt(dr_sq); //distance

    if (mdr >  EPSILON){ 

        delr = mdr - r0; //distance - r_eq
        mf = 2.0 * k_spring * delr/mdr;

        dU = delr * delr * k_spring;

        for (int j = 0; j < D; j++){
            f[ind*D + j] -= mf * dr_arr[j];
            f[nid*D + j] += mf * dr_arr[j];

            d_VirArr[list_ind * 5 + j] = dr_arr[j];
        }

        d_VirArr[list_ind * 5 + 3] = dU - e_bond;
        d_VirArr[list_ind * 5 + 4] = mf;
    }
    else{
        for (int j = 0; j < D; j++)
        { 
            d_VirArr[list_ind * 5 + j] = 0.0;
        }

        d_VirArr[list_ind * 5 + 3] = -e_bond;
        d_VirArr[list_ind * 5 + 4] = 0.0;
    }
}

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
    float r_n,
    float qind,
    float *L,
    float *Lh,
    int D,
    float* d_charges)
{

    int tmp_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (tmp_ind >= n_free)
        return;


    int list_ind = d_FREE[tmp_ind];
    int ind = d_index[list_ind];

    curandState l_state;
    l_state = d_states[ind];
    d_states[ind] = l_state;

    int lnid;
    int c = d_RN_ARRAY_COUNTER[list_ind];

    if (c != 0){
        l_state = d_states[ind];
        int r = (int)((curand_uniform(&l_state) * (INT_MAX + .999999)));
        d_states[ind] = l_state;
        lnid = d_RN_ARRAY[list_ind * ad_hoc_density * nncells + r%c];
    }
    else{
        return;
        }

    if (atomicCAS(&d_BONDS.get()[lnid * 2], 0, -1) == 0){ //lock the particle to bond with

        double dr_sq = 0.0;
        double dr0 = 0.0;
        double dr_arr[3];
        double delr = 0.0;
        double dU = 0.0;


        int nid = d_index[lnid];

        curandState l_state;
        l_state = d_states[ind];
        float rnd = curand_uniform(&l_state);
        d_states[ind] = l_state;

        for (int j = 0; j < D; j++){

            dr0 = x[ind * D + j] - x[nid * D + j];
            if (dr0 >  Lh[j]){dr_arr[j] = -1.0 * (L[j] - dr0);}
            else if (dr0 < -1.0 * Lh[j]){dr_arr[j] = (L[j] + dr0);}
            else{dr_arr[j] = dr0;}
            dr_sq += dr_arr[j] * dr_arr[j];
        }

        double mdr = sqrt(dr_sq); //distance
        if (mdr > EPSILON){ 
            delr = mdr - r0; //distance - r_eq
            dU = delr * delr * k_spring;
        }
        else
        {
            dU = 0.0;
            mdr = 0.0;
        }

        if (mdr <= r_n && rnd < exp(-dU + e_bond))
        {
            atomicExch(&d_BONDS.get()[list_ind * 2], 1);
            atomicExch(&d_BONDS.get()[lnid * 2], 1);

            atomicExch(&d_BONDS.get()[list_ind * 2 + 1], lnid);
            atomicExch(&d_BONDS.get()[lnid * 2 + 1], list_ind);

            if (qind != 0){
                d_charges[ind] -= qind;
                d_charges[nid] += qind;
            }
        }
        else
        {

            atomicExch(&d_BONDS.get()[list_ind * 2], 0);
            atomicExch(&d_BONDS.get()[lnid * 2], 0);

            atomicExch(&d_BONDS.get()[list_ind * 2 + 1], -1);
            atomicExch(&d_BONDS.get()[lnid * 2 + 1], -1);

        }
    } // if particle got locked

    __syncthreads();   
}



__global__ void d_break_bonds(
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
    float* d_charges)


{
    int tmp_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (tmp_ind >= n_bonded)
        return;

    int list_ind = d_BONDED[tmp_ind];
    int ind = d_index[list_ind];

    // printf("My name is %d I want to break free\n",ind);

    curandState l_state;
    l_state = d_states[ind];
    float rnd = curand_uniform(&l_state);
    d_states[ind] = l_state;

    int lnid = d_BONDS[list_ind * 2 + 1];
    int nid = d_index[lnid];

    double dr_sq = 0.0;
    double dr0;
    double dr_arr[3];
    double delr;
    double dU;


    for (int j = 0; j < D; j++){

        dr0 = x[ind * D + j] - x[nid * D + j];
        if (dr0 >  Lh[j]){dr_arr[j] = -1.0 * (L[j] - dr0);}
        else if (dr0 < -1.0 * Lh[j]){dr_arr[j] = (L[j] + dr0);}
        else{dr_arr[j] = dr0;}
        dr_sq += dr_arr[j] * dr_arr[j];
    }

    double mdr = sqrt(dr_sq); //distance
    if (mdr > EPSILON){ 
        delr = mdr - r0; //distance - r_eq
        dU = delr * delr * k_spring;
    }
    else
    {
        dU = 0.0;
    }

    if (rnd <= exp(dU - e_bond))
    {
        atomicExch(&d_BONDS.get()[list_ind * 2], 0);
        atomicExch(&d_BONDS.get()[lnid * 2], 0);

        atomicExch(&d_BONDS.get()[list_ind * 2 + 1], -1);
        atomicExch(&d_BONDS.get()[lnid * 2 + 1], -1);
    }
    __syncthreads();   
}

void Lewis::WriteBonds(void)
{

    // int flag = 0;

    this->BONDS = d_BONDS;
    ofstream bond_file;
    bond_file.open(file_name, ios::out | ios::app);

    // for (int j = 0; j < group->nsites; ++j){
    //     if (BONDS[2 * j + 1] != -1)
    //     {
    //         flag = 1;
    //         break;
    //     }
    // }

    bond_file << "TIMESTEP: " << step << " " << n_bonded << " " << n_free << " " << n_free + n_bonded << endl;
    for (int j = 0; j < group->nsites; ++j)
    {
        if (BONDS[2 * j + 1] != -1 && nlist->AD[j] == 1)
        {
            bond_file << group->index[j] + 1 << " " << this->group->index[BONDS[2 * j + 1]] + 1 << endl;
        }
    }
    bond_file.close();

}


void Lewis::UpdateVirial(void){

    VirArr = d_VirArr;

    // VirArr [5 * group size]
    // d_BONDED: stores group indices (not global index) of the bonded DONOR particles
    // Columns 0-2 displacement vector from the DONOR to the ACCEPTOR
    // Column 3 stores the current  bond energy
    // Column 4 stores mf
    // d_VirArr is updated in UpdateForces or make_bonds routines

    for (int k = 0; k < d_BONDED.size(); ++k){
        int j = d_BONDED[k];
        Udynamicbond += VirArr[j*5+3];
        float mf = VirArr[j*5+4];

        bondVir[0] += -mf * VirArr[j*5] * VirArr[j*5];
        bondVir[1] += -mf * VirArr[j*5+1] * VirArr[j*5+1];
        if ( Dim == 2 )
            bondVir[2] += -mf * VirArr[j*5] * VirArr[j*5+1];
        else if (Dim == 3)
        {
            bondVir[2] += -mf * VirArr[j*5+2] * VirArr[j*5+2];
            bondVir[3] += -mf * VirArr[j*5] * VirArr[j*5+1];
            bondVir[4] += -mf * VirArr[j*5] * VirArr[j*5+2];
            bondVir[5] += -mf * VirArr[j*5+1] * VirArr[j*5+2];
        }
    }
}
