// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


    // cuda_collect_x();

#include "Extraforce_lewis_cpu.h"
#include "Extraforce_dynamic.h"
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

#include <cstdlib>
#include <iostream>
#include <time.h>


// random_ind = std::rand() % (n_free);
// random_ind = std::rand() % (n_bonded);

#define EPSILON 1.0e-10

using namespace std;

LewisCPU::~LewisCPU() { return; }

LewisCPU::LewisCPU(istringstream &iss) : ExtraForce(iss)
{

    DynamicBonds.push_back(this);


    readRequiredParameter(iss, nlist_name);
	nlist_index = get_nlist_id(nlist_name);
    nlist = dynamic_cast<NListBonding*>(NLists.at(nlist_index));

    cout << "LewisCPU bonds active!" << endl;
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
    cout << "bond_log_freq: " << bond_log_freq << endl;
    readRequiredParameter(iss, file_name);
    cout << "output_file: " << file_name << endl;
    readRequiredParameter(iss, offset);
    cout << "offset: " << offset << endl;

    iss >> ramp_string;
    if (ramp_string == "ramp"){
        std::cout << "Energy ramp activated!" << std::endl;

        iss >> e_bond_final;
        iss >> ramp_reps;
        iss >> ramp_t_end;

        ramp_interval = ceil(float(ramp_t_end)/float(ramp_reps));
        ramp_counter = 0;
        RAMP_FLAG = 1;
        delta_e_bond = (e_bond_final - e_bond)/float(ramp_reps);
        std::cout << "Final energy: "<< e_bond_final <<", in " << ramp_reps <<" intervals of " << ramp_interval << " time steps" << std::endl;
    }
    else{
        RAMP_FLAG = 0;
    }

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

    d_BONDED.resize( group->nsites);
    d_FREE.resize( group->nsites);

    BONDED.resize( group->nsites);
    FREE.resize( group->nsites);


    for (int i = 0; i < nlist->n_donors; ++i){
        FREE[i] = nlist->d_DONORS[i];
        BONDED[i] = -1;
    }

    n_bonded = 0;
    n_free = nlist->n_donors;

    d_FREE = FREE;
    d_BONDED = BONDED;

    GRID = ceil(((float)nlist->n_donors)/float(threads));

}


void LewisCPU::AddExtraForce()
{   

    if (RAMP_FLAG == 1 && ramp_counter < ramp_reps && step % ramp_interval == 0 && step > 0){
        e_bond += delta_e_bond;
        std::cout << "At step: " << step <<" increased e_bond to: " << e_bond << std::endl;
        ++ramp_counter;
    }


    if (step % bond_freq == 0 && step >= bond_freq){

        int rnd = random()%2; //decide move sequence

        // First variant

        if (rnd == 0){

            /* Make Bonds */

            if (n_free > 0){

                MasterCharge->CalcCharges();
                MasterCharge->CalcEnergy();
                U_Electro_old = MasterCharge->energy;

                // Fields recalculated internally

                // make_bonds_cpu(x,f,
                // BONDS, nlist->RN_ARRAY, nlist->RN_ARRAY_COUNTER,
                // FREE, VirArr, 
                // nlist->nncells, nlist->ad_hoc_density,
                // group->index, group->nsites,
                // k_spring, e_bond, r0, nlist->r_n, qind,
                // d_L, d_Lh, Dim, d_charges);

                make_bonds_cpu(*x, *f, d_L, d_Lh, Dim, charges);

                // Copy device vectors to host vectors

                // std::cout << "B! Donor: "<< lewis_vect[0] << ", acceptor: " << lewis_vect[1] << ", P: " << lewis_vect[2] << ", dU_spring: " << dU_lewis[0] << ", old El: " << U_Electro_old <<  std::endl;

                    
                if (lewis_proceed == 1){
                    std::cout << "Acepted! Udating host charges" << std::endl;

                    // Update charges on the device

                    cudaMemcpy(d_charges, charges, ns * sizeof(float), cudaMemcpyHostToDevice);

                }
                else{

                    prepareDensityFields();
                    MasterCharge->CalcCharges();
                    MasterCharge->CalcEnergy();
                    std::cout << "Rejected! Recalculating fields!" << std::endl;
                }
            }

                n_bonded = 0;
                n_free = 0;

                d_BONDS = BONDS;
                for (int i = 0; i < group->nsites; ++i){
                    if (nlist->AD[i] == 1 && BONDS[2*i] == 1){
                        BONDED[n_bonded++] = i;
                    }
                    else if (nlist->AD[i] == 1 && BONDS[2*i] == 0){
                        FREE[n_free++] = i;
                    }
                }
                d_BONDED = BONDED;
                d_FREE = FREE;
            } 


            /*  Break bonds */ 
            
            if (n_bonded > 0){ 

                MasterCharge->CalcCharges();
                MasterCharge->CalcEnergy();
                U_Electro_old = MasterCharge->energy;

                // zero energy difference

                dU_lewis = 0.0f;
                ind1 = ind2 = lewis_proceed = -1;

                // Pick random particle to bond

                random_ind = std::rand() % (n_bonded);

                // Pick random particle to bond

                d_break_bonds_lewis_full_1<<<1, 1>>>(d_x,d_f,
                    d_BONDS.data(),
                    nlist->d_RN_ARRAY.data(), nlist->d_RN_ARRAY_COUNTER.data(),
                    d_BONDED.data(), d_VirArr.data(), random_ind,
                    nlist->nncells, nlist->ad_hoc_density,
                    group->d_index.data(), group->nsites, d_states,
                    k_spring, e_bond, r0, nlist->r_n, qind, d_L, d_Lh, Dim,d_charges,
                    d_lewis_vect.data(), d_dU_lewis.data());


                // Copy device vectors to host vectors
                lewis_vect = d_lewis_vect;
                dU_lewis = d_dU_lewis;
                

                // std::cout << "A! Acceptor: "<< lewis_vect[0] << ", donor: " << lewis_vect[1] << ", P: " << lewis_vect[2] << ", dU_spring: " << dU_lewis[0] << ", old El: " << U_Electro_old <<  std::endl;

                prepareDensityFields();
                MasterCharge->CalcCharges();
                MasterCharge->CalcEnergy();


                // Update energy

                float dUEl = U_Electro_old - MasterCharge->energy;

                // std::cout << "Int: " << MasterCharge->energy << ", dUE: " << dUEl<< std::endl;

                dU_lewis[0] += dUEl;
                d_dU_lewis = dU_lewis;

                

                d_break_bonds_lewis_full_2<<<1, 1>>>(d_x,d_f,
                    d_BONDS.data(),
                    nlist->d_RN_ARRAY.data(), nlist->d_RN_ARRAY_COUNTER.data(),
                    d_BONDED.data(), d_VirArr.data(), random_ind,
                    nlist->nncells, nlist->ad_hoc_density,
                    group->d_index.data(), group->nsites, d_states,
                    k_spring, e_bond, r0, nlist->r_n, qind, d_L, d_Lh, Dim,d_charges,
                    d_lewis_vect.data(), d_dU_lewis.data());

                prepareDensityFields();
                MasterCharge->CalcCharges();
                MasterCharge->CalcEnergy();

                d_lewis_vect = lewis_vect;

                // std::cout << "dU_new: " << d_dU_lewis[0] <<  ", E_old: " << U_Electro_old << ", E_new: " << MasterCharge->energy << std::endl;

                // Update host charges
                cudaMemcpy(charges, d_charges, ns * sizeof(float), cudaMemcpyDeviceToHost);

                // if (d_lewis_vect[2] == 1){
                //     std::cout << "Accepted bond break!" << std::endl;
                // }
                // else{
                //     std::cout << "Rejected bond break!" << std::endl;
                // }

                n_bonded = 0;
                n_free = 0;

                BONDS = d_BONDS;
                for (int i = 0; i < group->nsites; ++i){
                    if (nlist->AD[i] == 1 && BONDS[2*i] == 1){
                        BONDED[n_bonded++] = i;
                    }
                    else if (nlist->AD[i] == 1 && BONDS[2*i] == 0){
                        FREE[n_free++] = i;
                    }
                }

                d_BONDED = BONDED;
                d_FREE = FREE;

            } 
        }
    } // end if (step % lewis_bond_freq == 0 && step >= bond_freq)

    if(step >= bond_freq){
        d_update_forces<<<GRID, threads>>>(d_x, d_f, d_L, d_Lh,
            k_spring, e_bond, r0,
            d_BONDS.data(), d_BONDED.data(), d_VirArr.data(), n_bonded,
            group->d_index.data(), group->nsites, Dim);
    }

    if (step == 0){
        const char* fname = file_name.c_str();
        remove(fname);
    }

    if (step % bond_log_freq == 0 && step >= offset)
    {
        LewisCPU::WriteBonds();
    }
}


/*
Updates forces acting on particles due to LewisCPU bonds
*/


                // make_bonds_cpu(x,f,
                // BONDS, nlist->RN_ARRAY, nlist->RN_ARRAY_COUNTER,
                // FREE, VirArr, 
                // nlist->nncells, nlist->ad_hoc_density,
                // group->index, group->nsites,
                // k_spring, e_bond, r0, nlist->r_n, qind,
                // d_L, d_Lh, Dim, d_charges);

void LewisCPU::make_bonds_cpu(
    const float *x,
    const float* f,    
    float *L,
    float *Lh,
    int D,
    float* charges)
{

    float dU_lewis = 0.0f;
    // id1 id2 proceed_flag
    int ind1 = -1;       
    int ind2 = -1;
    int lewis_proceed = -1;

    // Pick random particle to bond
    int tmp_ind = std::rand() % (n_free);
    int list_ind = FREE[tmp_ind];
    int ind = group->index[list_ind];

    int lnid = -1;
    int c = RN_ARRAY_COUNTER[list_ind];

    if (c != 0){
        int r = std::rand();
        lnid = RN_ARRAY[list_ind * ad_hoc_density * nncells + r%c];
    }
    else{ return;}


    if (BONDS[lnid * 2] == 0){ //if particle is non-bonded

        double dr_sq = 0.0;
        double dr0 = 0.0;
        double dr_arr[3];
        double delr = 0.0;
        double dU = 0.0;


        int nid = index[lnid];


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

        dU_lewis += dU;

        if (mdr <= r_n){

            printf("Ind Nid ch1 ch2: %d %d %f %f\n", ind,nid, charges[ind],charges[nid]);    
            charges[ind] += qind; 
            charges[nid] -= qind;
            printf("Ind Nid ch1 ch2: %d %d %f %f\n", ind,nid, charges[ind],charges[nid]);   

        }

        // Distance exceeds the permitted one - aborting the attempt
        else{
            atomicExch(&BONDS[list_ind * 2], 0);
            atomicExch(&BONDS[lnid * 2], 0);

            BONDS[list_ind * 2 + 1] = -1;
            BONDS[lnid * 2 + 1] = -1;
            return;
        }


        // Recalculate electrostatic field

        prepareDensityFields();
        MasterCharge->CalcCharges();
        MasterCharge->CalcEnergy();


        float dUEl = U_Electro_old - MasterCharge->energy;

        std::cout << "Old El:" << U_Electro_old << ", current E: " << MasterCharge->energy << std::endl;

        dU_lewis += dUEl;



    }
    else{
        return;
    }

    // // part 2.

    // int list_ind = d_lewis_vect[0];
    // int lnid = d_lewis_vect[1];

    // int ind = d_index[list_ind];
    // int nid = d_index[lnid];


    // curandState l_state;
    // l_state = d_states[ind];
    // float rnd = curand_uniform(&l_state);
    // d_states[ind] = l_state;


    // d_lewis_vect[2] = -1;

    // if (rnd < exp(-d_dU_lewis[0] + e_bond))
    // {
    //     d_lewis_vect[2] = 1;
    //     atomicExch(&d_BONDS.get()[list_ind * 2], 1);
    //     atomicExch(&d_BONDS.get()[lnid * 2], 1);

    //     atomicExch(&d_BONDS.get()[list_ind * 2 + 1], lnid);
    //     atomicExch(&d_BONDS.get()[lnid * 2 + 1], list_ind);
    // }

    // else
    // {


    //     atomicExch(&d_BONDS.get()[list_ind * 2], 0);
    //     atomicExch(&d_BONDS.get()[lnid * 2], 0);

    //     atomicExch(&d_BONDS.get()[list_ind * 2 + 1], -1);
    //     atomicExch(&d_BONDS.get()[lnid * 2 + 1], -1);

    //     // Fix charges  
    //     d_charges[ind] -= qind; 
    //     d_charges[nid] += qind;

    // }
    return;
}



/* Break bonds */


void break_bonds_lewis_full_1(
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
    thrust::device_ptr<int> d_lewis_vect,
    thrust::device_ptr<float> d_dU_lewis)

{
    int tmp_ind = random_id;
    int list_ind = d_BONDED[tmp_ind];
    int ind = d_index[list_ind];

    int lnid = d_BONDS[list_ind * 2 + 1];
    int nid = d_index[lnid];

    d_lewis_vect[0] = list_ind;
    d_lewis_vect[1] = lnid;

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

    d_dU_lewis[0] = dU;

    // Temporary charge change
    d_charges[ind] -= qind; 
    d_charges[nid] += qind;

}


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
    thrust::device_ptr<int> d_lewis_vect,
    thrust::device_ptr<float> d_dU_lewis)

{


    int list_ind = d_lewis_vect[0];
    int lnid = d_lewis_vect[1];

    int ind = d_index[list_ind];
    int nid = d_index[lnid];


    curandState l_state;
    l_state = d_states[ind];
    float rnd = curand_uniform(&l_state);
    d_states[ind] = l_state;


    if (rnd <= exp(d_dU_lewis[0] - e_bond))
    {
        atomicExch(&d_BONDS.get()[list_ind * 2], 0);
        atomicExch(&d_BONDS.get()[lnid * 2], 0);

        atomicExch(&d_BONDS.get()[list_ind * 2 + 1], -1);
        atomicExch(&d_BONDS.get()[lnid * 2 + 1], -1);
        d_lewis_vect[2] = 1;
        // printf("Accepted bond break!");

    }
    else{

        // Restore bound charges
        d_charges[ind] += qind; 
        d_charges[nid] -= qind;
        // printf("Rejected bond break!");

    }
}

void LewisCPU::WriteBonds(void)
{

    this->BONDS = d_BONDS;
    ofstream bond_file;
    // bond_file.open(file_name, ios::out | ios::app);


    // bond_file << "TIMESTEP: " << global_step << " " << n_bonded << " " << n_free << " " << n_free + n_bonded << endl;
    // for (int j = 0; j < group->nsites; ++j)
    // {
    //     if (BONDS[2 * j + 1] != -1 && nlist->AD[j] == 1)
    //     {
    //         bond_file << group->index[j] + 1 << " " << this->group->index[BONDS[2 * j + 1]] + 1 << endl;
    //     }
    // }
    // bond_file.close();

    bond_file.open("bond_data", ios::out | ios::app);

    bond_file << global_step << " " << float(n_bonded)/float(n_free + n_bonded) << " " << MasterCharge->energy << endl;
    bond_file.close();

}


void LewisCPU::UpdateVirial(void){

    VirArr = d_VirArr;

    // VirArr [5 * group size]
    // d_BONDED: stores group indices (not global index) of the bonded DONOR particles
    // Columns 0-2 displacement vector from the DONOR to the ACCEPTOR
    // Column 3 stores the current  bond energy
    // Column 4 stores mf
    // d_VirArr is updated in UpdateForces or make_bonds routines

    for (int k = 0; k < n_bonded; ++k){
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
