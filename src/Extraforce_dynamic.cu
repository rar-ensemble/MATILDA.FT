// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


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


#define EPSILON 1.0e-10

using namespace std;


Dynamic::~Dynamic() { return; }

Dynamic::Dynamic(istringstream &iss) : ExtraForce(iss)
{


    DynamicBonds.push_back(this);
    thrust::default_random_engine g(time(0));


    cout << "Dynamic bonds active!" << endl;

    readRequiredParameter(iss, ad_file); //file that maps acceptors and donors to group indices
    cout << "AD mapping file: " << ad_file << endl; 


    // Read the AD mapping file and assign correct indices
    //ACCEPTORS array stores group indices of acceptors, and vice versa for donors

    std::string line2;
    ifstream in2(ad_file);
    getline(in2, line2);
    istringstream str2(line2);
    

    std::vector<int> d_vect, a_vect;
    int adc = 0;
    while (str2 >> adc){
        if (adc == 1){
            d_vect.push_back(1); // stores indices within the group
        }

        else if(adc == 0){
            a_vect.push_back(1);
        }
    }
    if (d_vect.size() >= a_vect.size()){
        donor_tag = 1;
        acceptor_tag = 0;
        std::cout << "Donors > Acceptors" << std::endl;
    }
    else{
        donor_tag = 0;
        acceptor_tag = 1;
        std::cout << "Acceptors > Donors" << std::endl;
    }

    in2.close();

    mbbond.resize(2);
    mbbond[0] = mbbond[1] = 0;
    d_mbbond = mbbond;
    

    std::string line;
    ifstream in(ad_file);
    getline(in, line);
    istringstream str(line);

    int ad = 0;
    int count = 0;
    while (str >> ad){
        AD.push_back(ad);
        if (ad == donor_tag){
            d_DONORS.push_back(count); // stores indices within the group
        }

        else if(ad == acceptor_tag){
            d_ACCEPTORS.push_back(count);
        }
        ++count;
    }

    in.close();
    

    n_donors = d_DONORS.size();
    n_acceptors = d_ACCEPTORS.size();


    ACCEPTORS = d_ACCEPTORS;
    DONORS = d_DONORS;

    S_ACCEPTORS.resize(n_donors * n_acceptors);


    for (int i = 0; i < n_donors; ++i){
        // RN_ARRAY_COUNTER.push_back(n_acceptors);
        thrust::shuffle(ACCEPTORS.begin(),ACCEPTORS.end(),g);
        for (int j = 0; j < ACCEPTORS.size(); ++j){
            S_ACCEPTORS[i * n_acceptors + j] = ACCEPTORS[j];
        }
    }

    d_S_ACCEPTORS = S_ACCEPTORS;

    // d_RN_ARRAY = RN_ARRAY;
    // d_RN_ARRAY_COUNTER = RN_ARRAY_COUNTER;


    readRequiredParameter(iss, k_spring);
    cout << "k_spring: " << k_spring << endl;
    readRequiredParameter(iss, e_bond);
    cout << "e_bond: " << e_bond << endl;
    readRequiredParameter(iss, r0);
    cout << "r0: " << r0 << endl;
    readRequiredParameter(iss, r_n);
     cout << "r_n: " << r_n << endl;
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
    cout << "Donors: " << n_donors << endl;
    cout << "Acceptors: " << n_acceptors << endl;



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

    d_BONDED.resize( n_donors);
    d_FREE.resize( n_donors);
    d_FREE_ACCEPTORS.resize(n_acceptors);

    BONDED.resize( n_donors);
    FREE.resize( n_donors);
    FREE_ACCEPTORS.resize(n_acceptors);


    for (int i = 0; i < n_donors; ++i){
        FREE[i] = d_DONORS[i];
        BONDED[i] = -1;
    }

    for (int i = 0; i < n_acceptors; ++i){
        FREE_ACCEPTORS[i] = d_ACCEPTORS[i];
    }

    n_bonded = 0;
    n_free_donors = n_donors;
    n_free_acceptors = n_acceptors;

    d_FREE = FREE;
    d_BONDED = BONDED;
    d_FREE_ACCEPTORS = FREE_ACCEPTORS;

    GRID = ceil(((float)n_donors)/float(threads));


}


void Dynamic::AddExtraForce()
{   

    if (RAMP_FLAG == 1 && ramp_counter < ramp_reps && step % ramp_interval == 0 && step > 0){
        e_bond += delta_e_bond;
        std::cout << "At step: " << step <<" increased e_bond to: " << e_bond << std::endl;
        ++ramp_counter;
    }



    if (step % bond_freq == 0 && step > 0){

        thrust::shuffle(d_ACCEPTORS.begin(),d_ACCEPTORS.end(),g);



        for (int i = 0; i < 2; ++i){

            int rnd = random()%2; //decide move sequence

            if (rnd == 0){

                if (n_free_donors > 0 && ((double) rand() / (RAND_MAX)) <=  2.0 * (float)n_acceptors/(n_donors + n_acceptors)){
                    d_make_bonds<<<GRID, threads>>>(d_x,d_f,
                        d_BONDS.data(),
                        d_ACCEPTORS.data(),
                        d_FREE.data(), d_VirArr.data(), n_free_donors,
                        n_donors,n_acceptors,
                        group->d_index.data(), group->nsites, d_states,
                        k_spring, e_bond, r0, r_n, d_mbbond.data(), d_L, d_Lh, Dim);


                    n_bonded = 0;
                    n_free_donors = 0;
                    n_free_acceptors = 0;

                    BONDS = d_BONDS;
                    for (int i = 0; i < group->nsites; ++i){
                        if (AD[i] == donor_tag && BONDS[2*i] == 1){
                            BONDED[n_bonded++] = i;
                        }
                        else if (AD[i] == donor_tag && BONDS[2*i] == 0){
                            FREE[n_free_donors++] = i;
                        }
                        else if (AD[i] == acceptor_tag && BONDS[2*i] == 0){
                         FREE_ACCEPTORS[n_free_acceptors++];
                        }
                    }

                    d_BONDED = BONDED;
                    d_FREE = FREE;
                    d_FREE_ACCEPTORS = FREE_ACCEPTORS;


                } 
                if (n_bonded > 0 && ((double) rand() / (RAND_MAX)) < float(n_bonded)/(n_donors + n_acceptors)*2.0){
 
                    
                    d_break_bonds<<<GRID, threads>>>(d_x,
                        d_BONDS.data(),
                        d_BONDED.data(),n_bonded,
                        n_donors,n_acceptors,
                        group->d_index.data(), group->nsites, d_states,
                        k_spring, e_bond, r0, d_mbbond.data(), d_L, d_Lh, Dim);


                    n_bonded = 0;
                    n_free_donors = 0;
                    n_free_acceptors = 0;

                    BONDS = d_BONDS;
                    for (int i = 0; i < group->nsites; ++i){
                        if (AD[i] == donor_tag && BONDS[2*i] == 1){
                            BONDED[n_bonded++] = i;
                        }
                        else if (AD[i] == donor_tag && BONDS[2*i] == 0){
                            FREE[n_free_donors++] = i;
                        }
                        else if (AD[i] == acceptor_tag && BONDS[2*i] == 0){
                         FREE_ACCEPTORS[n_free_acceptors++];
                        }
                    }

                    d_BONDED = BONDED;
                    d_FREE = FREE;
                    d_FREE_ACCEPTORS = FREE_ACCEPTORS;

                } 
            }
            else{

                if (n_bonded > 0 && ((double) rand() / (RAND_MAX)) < float(n_bonded)/(n_donors + n_acceptors)*2.0){
                    d_break_bonds<<<GRID, threads>>>(d_x,
                        d_BONDS.data(),
                        d_BONDED.data(),n_bonded,
                        n_donors,n_acceptors,
                        group->d_index.data(), group->nsites, d_states,
                        k_spring, e_bond, r0, d_mbbond.data(), d_L, d_Lh, Dim);


                    n_bonded = 0;
                    n_free_donors = 0;
                    n_free_acceptors = 0;

                    BONDS = d_BONDS;
                    for (int i = 0; i < group->nsites; ++i){
                        if (AD[i] == donor_tag && BONDS[2*i] == 1){
                            BONDED[n_bonded++] = i;
                        }
                        else if (AD[i] == donor_tag && BONDS[2*i] == 0){
                            FREE[n_free_donors++] = i;
                        }
                        else if (AD[i] == acceptor_tag && BONDS[2*i] == 0){
                         FREE_ACCEPTORS[n_free_acceptors++];
                        }
                    }

                    d_BONDED = BONDED;
                    d_FREE = FREE;
                    d_FREE_ACCEPTORS = FREE_ACCEPTORS;


                } // if n_free_donors > 0

                if (n_free_donors > 0 && ((double) rand() / (RAND_MAX)) <=  2.0 * (float)n_acceptors/(n_donors + n_acceptors)){
                        d_make_bonds<<<GRID, threads>>>(d_x,d_f,
                            d_BONDS.data(),
                            d_ACCEPTORS.data(),
                            d_FREE.data(), d_VirArr.data(), n_free_donors,
                            n_donors,n_acceptors,
                            group->d_index.data(), group->nsites, d_states,
                            k_spring, e_bond, r0, r_n, d_mbbond.data(), d_L, d_Lh, Dim);

                                // update the bonded array

                    n_bonded = 0;
                    n_free_donors = 0;
                    n_free_acceptors = 0;

                    BONDS = d_BONDS;
                    for (int i = 0; i < group->nsites; ++i){
                        if (AD[i] == donor_tag && BONDS[2*i] == 1){
                            BONDED[n_bonded++] = i;
                        }
                        else if (AD[i] == donor_tag && BONDS[2*i] == 0){
                            FREE[n_free_donors++] = i;
                        }
                        else if (AD[i] == acceptor_tag && BONDS[2*i] == 0){
                         FREE_ACCEPTORS[n_free_acceptors++];
                        }
                    }

                    d_BONDED = BONDED;
                    d_FREE = FREE;
                    d_FREE_ACCEPTORS = FREE_ACCEPTORS;


                    }
            } // option 2
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
        Dynamic::WriteBonds();
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
    thrust::device_ptr<int> d_ACCEPTORS,
    thrust::device_ptr<int> d_FREE,
    thrust::device_ptr<float> d_VirArr,
    int n_free_donors,
    int n_donors,
    int n_acceptors,
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
    int D)
{

    int tmp_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (tmp_ind >= n_free_donors)
        return;

    int n_bonded = n_donors - n_free_donors;
    int n_free_acceptors = n_acceptors - n_bonded;

    int list_ind = d_FREE[tmp_ind];
    int ind = d_index[list_ind];

    // int max_counter = (float)n_acceptors/(L[0]*L[1]*L[2]) * (4.0/3.0 * 3.1415926 * r_n*r_n*r_n);


    curandState l_state;
    l_state = d_states[ind];
    float rnd = curand_uniform(&l_state);
    d_states[ind] = l_state;
  
    if (rnd > 0.20){return;} // only run on 1% of particles to avoid excessive fluctuations

    l_state = d_states[ind];
    rnd = curand_uniform(&l_state);
    d_states[ind] = l_state;


    int lnid;
    int c = n_acceptors-1;



    double dr_sq = 0.0;
    double dr0 = 0.0;
    double dr_arr[3];
    double delr = r_n + 1.0;
    double dU = 0.0;
    int nid;
    int r;


    while (delr > r_n){


        dr_sq = 0.0;
        dr0 = 0.0;
        dU = 0.0;

        l_state = d_states[ind];
        r = (int)round((curand_uniform(&l_state) * c));
        d_states[ind] = l_state;
        lnid = d_ACCEPTORS[r%n_acceptors];


        // atomicAdd(&d_mbbond.get()[0],1);

        nid = d_index[lnid];



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
            delr = 0.0;
        }
    }


    if (atomicCAS(&d_BONDS.get()[lnid * 2], 0, -1) == 0){ //lock the particle to bond with

        curandState l_state;
        l_state = d_states[ind];
        float rnd = curand_uniform(&l_state);
        d_states[ind] = l_state;

        if (rnd < exp(-dU + e_bond))
        {
            atomicExch(&d_BONDS.get()[list_ind * 2], 1);
            atomicExch(&d_BONDS.get()[lnid * 2], 1);

            atomicExch(&d_BONDS.get()[list_ind * 2 + 1], lnid);
            atomicExch(&d_BONDS.get()[lnid * 2 + 1], list_ind);

        }
        else
        {

            atomicExch(&d_BONDS.get()[list_ind * 2], 0);
            atomicExch(&d_BONDS.get()[lnid * 2], 0);

            atomicExch(&d_BONDS.get()[list_ind * 2 + 1], -1);
            atomicExch(&d_BONDS.get()[lnid * 2 + 1], -1);

        }
    } // if particle got locked
}



__global__ void d_break_bonds(
    const float *x,
    thrust::device_ptr<int> d_BONDS,
    thrust::device_ptr<int> d_BONDED,
    int n_bonded,
    int n_donors,
    int n_acceptors,
    thrust::device_ptr<int> d_index, 
    const int ns,        
    curandState *d_states,
    float k_spring,
    float e_bond,
    float r0,
    thrust::device_ptr<int> d_mbbond,
    float *L,
    float *Lh,
    int D)


{
    int tmp_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (tmp_ind >= n_bonded)
        return;

    int list_ind = d_BONDED[tmp_ind];
    int ind = d_index[list_ind];


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

    int n_free_acceptors = n_acceptors - n_bonded;
    int n_free_donors = n_donors-n_bonded;



    if (rnd > 0.20){return;} // only run on 1% of particles to avoid excessive fluctuations


    l_state = d_states[ind];
    rnd = curand_uniform(&l_state);
    d_states[ind] = l_state;


    // atomicAdd(&d_mbbond.get()[1],1);


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
        delr = 0.0;
    }


    if (rnd <= exp(dU - e_bond))
    {
        atomicExch(&d_BONDS.get()[list_ind * 2], 0);
        atomicExch(&d_BONDS.get()[lnid * 2], 0);

        atomicExch(&d_BONDS.get()[list_ind * 2 + 1], -1);
        atomicExch(&d_BONDS.get()[lnid * 2 + 1], -1);
    }
}

void Dynamic::WriteBonds(void)
{

    // int flag = 0;

    this->BONDS = d_BONDS;
    mbbond = d_mbbond;
    ofstream bond_file;
    bond_file.open(file_name, ios::out | ios::app);

    bond_file << global_step << " " << float(n_bonded)/min(n_donors,n_acceptors) << std::endl;//" " <<mbbond[0] <<" " << mbbond[1] << endl;
    // bond_file << "TIMESTEP: " << global_step << " " << n_bonded << " " << n_free_donors << " " << n_free_donors + n_bonded << endl;
    // for (int j = 0; j < group->nsites; ++j)
    // {
    //     if (BONDS[2 * j + 1] != -1 && AD[j] == acceptor_tag)
    //     {
    //         bond_file << group->index[j] + 1 << " " << this->group->index[BONDS[2 * j + 1]] + 1 << endl;
    //     }
    // }
    bond_file.close();

}


void Dynamic::UpdateVirial(void){

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
