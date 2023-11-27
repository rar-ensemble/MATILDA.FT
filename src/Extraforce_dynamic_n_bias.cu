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





    readRequiredParameter(iss, k_spring);
    cout << "k_spring: " << k_spring << endl;
    readRequiredParameter(iss, e_bond);
    cout << "e_bond: " << e_bond << endl;
    readRequiredParameter(iss, r0);
    cout << "r0: " << r0 << endl;
    readRequiredParameter(iss, r_n);
    cout << "r_n: " << r_n << endl;
    readRequiredParameter(iss, sticker_density);
    cout << "sticker_density: " << sticker_density << endl;
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
    thrust::shuffle(FREE_ACCEPTORS.begin(),FREE_ACCEPTORS.begin() + n_free_acceptors,g);
    d_FREE = FREE;
    d_BONDED = BONDED;
    d_FREE_ACCEPTORS = FREE_ACCEPTORS;

    GRID = ceil(((float)n_donors)/float(threads));

    // Prepare neighbour-list

    int tmp;
    xyz = 1;
    int resize_flag = 0;

    for (int i = 0; i < Dim; i++)
    {
        tmp = floor(L[i]/r_n);
        Nxx.push_back(tmp);
        xyz *= int(tmp);
    }

    for (auto i: Nxx){
        if (i < 3){
            resize_flag = 1;
        }
    }

    if (resize_flag == 1){
        std::cout << "New r_n! N-list spans entire box!" << std::endl;
        if (Dim==3){
            xyz = 27;
            Nxx[0] = Nxx[1] = Nxx[2] = 3;
            r_n = max(L[0],L[1]);
            r_n = max(r_n, L[3]);
        }
        else{
            xyz = 9;
            Nxx[0] = Nxx[1] = 3;
            r_n = max(L[0],L[1]);
        }

    }

    for (int i = 0; i < Dim; i++)
    {
        Lg.push_back(L[i] / float(Nxx[i]));
    }
    for (int i = 0; i < Dim; i++)
    {
        std::cout << "Nx[" << i << "]: " << Nxx[i] << " |L:" << L[i] << " |dL: " << Lg[i] << endl;
    }


    if (Dim == 2){Nxx.push_back(1); Lg[2] = 1.0;}

    d_Nxx.resize(3);
    d_Nxx = Nxx;
    d_Lg.resize(3);
    d_Lg = Lg;

    if (Dim == 3){nncells = 27;}
    if (Dim == 2){nncells = 9;}         
    

    // Grid counters

    d_MASTER_GRID.resize(xyz * sticker_density);                 
    d_MASTER_GRID_counter.resize(xyz);

    // (R-cutoff, n-list) counters

    d_RN_ARRAY.resize(group->nsites * sticker_density * nncells);
    d_RN_ARRAY_COUNTER.resize(group->nsites);


    d_LOW_DENS_FLAG.resize(group->nsites);

    thrust::fill(d_MASTER_GRID.begin(),d_MASTER_GRID.end(),-1);
    thrust::fill(d_MASTER_GRID_counter.begin(),d_MASTER_GRID_counter.end(),0);

    thrust::fill(d_RN_ARRAY.begin(),d_RN_ARRAY.end(),-1);
    thrust::fill(d_RN_ARRAY_COUNTER.begin(),d_RN_ARRAY_COUNTER.end(),0);

    thrust::fill(d_LOW_DENS_FLAG.begin(), d_LOW_DENS_FLAG.end(), 0);

    std::cout << "Total cells: " << xyz << ", sticker_density: " << sticker_density << ", Dim: " << Dim << ", n_cells: " << nncells << endl;


}


void Dynamic::AddExtraForce()
{   

    // if (RAMP_FLAG == 1 && ramp_counter < ramp_reps && step % ramp_interval == 0 && step > 0){
    //     e_bond += delta_e_bond;
    //     std::cout << "At step: " << step <<" increased e_bond to: " << e_bond << std::endl;
    //     ++ramp_counter;
    // }



    if (step % bond_freq == 0 && step > 0){


        thrust::fill(d_MASTER_GRID.begin(),d_MASTER_GRID.end(),-1);
        thrust::fill(d_MASTER_GRID_counter.begin(),d_MASTER_GRID_counter.end(),0);

        thrust::fill(d_RN_ARRAY.begin(),d_RN_ARRAY.end(),-1);
        thrust::fill(d_RN_ARRAY_COUNTER.begin(),d_RN_ARRAY_COUNTER.end(),0);

        thrust::fill(d_LOW_DENS_FLAG.begin(), d_LOW_DENS_FLAG.end(), 0);


        d_update_grid<<<GRID, threads>>>(d_x, d_Lh, d_L,
            d_MASTER_GRID_counter.data(), d_MASTER_GRID.data(),
            d_Nxx.data(), d_Lg.data(),
            d_LOW_DENS_FLAG.data(),
            d_ACCEPTORS.data(),
            nncells, n_acceptors, sticker_density,
            group->d_index.data(), group->nsites, Dim);


        // Updates n-list for the donors

        d_update_neighbours<<<GRID, threads>>>(d_x, d_Lh, d_L,
            d_BONDS.data(),
            d_MASTER_GRID_counter.data(), d_MASTER_GRID.data(),
            d_Nxx.data(), d_Lg.data(),
            d_RN_ARRAY.data(), d_RN_ARRAY_COUNTER.data(),
            d_DONORS.data(),
            nncells, n_donors, r_n, sticker_density,
            group->d_index.data(), group->nsites, Dim);


        // Updates the distribution of acceptors on the grid

        int sum = thrust::reduce(d_LOW_DENS_FLAG.begin(), d_LOW_DENS_FLAG.end(), 0, thrust::plus<int>());
        LOW_DENS_FLAG = float(sum)/float(d_MASTER_GRID_counter.size());
        if (LOW_DENS_FLAG > 0){
            
            cout << "Input density was: " << sticker_density <<" but at least "<< sticker_density + LOW_DENS_FLAG <<" is required"<<endl;
            die("Low sticker_density!");
        }
        
        RN_ARRAY = d_RN_ARRAY;
        RN_ARRAY_COUNTER = d_RN_ARRAY_COUNTER;


        for (int i = 0; i < 2; ++i){

            int rnd = random()%2; //decide move sequence

            if (rnd == 0){

                if (n_free_donors > 0){
                    d_make_bonds<<<GRID, threads>>>(d_x,d_f,
                        d_BONDS.data(),
                        d_FREE_ACCEPTORS.data(),
                        d_FREE.data(),
                        d_RN_ARRAY.data(),d_RN_ARRAY_COUNTER.data(),
                        d_VirArr.data(), n_free_donors,
                        n_donors,n_acceptors,
                        sticker_density,nncells,
                        group->d_index.data(), group->nsites, d_states,
                        k_spring, e_bond, r0, r_n, d_mbbond.data(), d_L, d_Lh, Dim);


                    n_bonded = 0;
                    n_free_donors = 0;
                    n_free_acceptors = 0;
                    
                    thrust::fill(FREE_ACCEPTORS.begin(),FREE_ACCEPTORS.end(),-1);

                    BONDS = d_BONDS;
                    for (int i = 0; i < group->nsites; ++i){
                        if (AD[i] == donor_tag && BONDS[2*i] == 1){
                            BONDED[n_bonded++] = i;
                        }
                        else if (AD[i] == donor_tag && BONDS[2*i] == 0){
                            FREE[n_free_donors++] = i;
                        }
                        else if (AD[i] == acceptor_tag && BONDS[2*i] == 0){
                         FREE_ACCEPTORS[n_free_acceptors++] = i;
                        }
                    }
                    thrust::shuffle(FREE_ACCEPTORS.begin(),FREE_ACCEPTORS.begin() + n_free_acceptors,g);

                    // for (auto element : FREE_ACCEPTORS){
                    //     std::cout << element << " ";
                    // }
                    // std::cout << std::endl << std::endl;

                    d_BONDED = BONDED;
                    d_FREE = FREE;
                    d_FREE_ACCEPTORS = FREE_ACCEPTORS;
 

                } 

                if (n_bonded > 0){
 
                    
                    d_break_bonds<<<GRID, threads>>>(d_x,
                        d_BONDS.data(),
                        d_BONDED.data(),n_bonded,
                        n_donors,n_acceptors, r_n,
                        group->d_index.data(), group->nsites, d_states,
                        k_spring, e_bond, r0, d_mbbond.data(), d_L, d_Lh, Dim);

                        thrust::fill(FREE_ACCEPTORS.begin(),FREE_ACCEPTORS.end(),-1);
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
                         FREE_ACCEPTORS[n_free_acceptors++] = i;
                        }
                    }
                    thrust::shuffle(FREE_ACCEPTORS.begin(),FREE_ACCEPTORS.begin() + n_free_acceptors,g);

                    d_BONDED = BONDED;
                    d_FREE = FREE;
                    d_FREE_ACCEPTORS = FREE_ACCEPTORS;


                } 
            }

            else{

                if (n_bonded > 0){
                    d_break_bonds<<<GRID, threads>>>(d_x,
                        d_BONDS.data(),
                        d_BONDED.data(),n_bonded,
                        n_donors,n_acceptors,
                        r_n,
                        group->d_index.data(), group->nsites, d_states,
                        k_spring, e_bond, r0, d_mbbond.data(), d_L, d_Lh, Dim);

                        thrust::fill(FREE_ACCEPTORS.begin(),FREE_ACCEPTORS.end(),-1);
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
                         FREE_ACCEPTORS[n_free_acceptors++] = i;
                        }
                    }
                    thrust::shuffle(FREE_ACCEPTORS.begin(),FREE_ACCEPTORS.begin() + n_free_acceptors,g);

                    d_BONDED = BONDED;
                    d_FREE = FREE;
                    d_FREE_ACCEPTORS = FREE_ACCEPTORS;
                    


                } // if n_free_donors > 0

                if (n_free_donors > 0){
                        d_make_bonds<<<GRID, threads>>>(d_x,d_f,
                            d_BONDS.data(),
                            d_FREE_ACCEPTORS.data(),
                            d_FREE.data(),
                            d_RN_ARRAY.data(),d_RN_ARRAY_COUNTER.data(),
                            d_VirArr.data(), n_free_donors,
                            n_donors,n_acceptors,
                            sticker_density,nncells,
                            group->d_index.data(), group->nsites, d_states,
                            k_spring, e_bond, r0, r_n, d_mbbond.data(), d_L, d_Lh, Dim);

                                // update the bonded array
                                thrust::fill(FREE_ACCEPTORS.begin(),FREE_ACCEPTORS.end(),-1);
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
                         FREE_ACCEPTORS[n_free_acceptors++] = i;
                        }
                    }
                    thrust::shuffle(FREE_ACCEPTORS.begin(),FREE_ACCEPTORS.begin() + n_free_acceptors,g);

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
    thrust::device_ptr<int> d_RN_ARRAY,
    thrust::device_ptr<int> d_RN_ARRAY_COUNTER,
    thrust::device_ptr<float> d_VirArr,
    int n_free_donors,
    int n_donors,
    int n_acceptors,
    int sticker_density,
    int nncells,
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
  
    if (rnd > 0.05){return;} // only run on 1% of particles to avoid excessive fluctuations

    l_state = d_states[ind];
    rnd = curand_uniform(&l_state);
    d_states[ind] = l_state;


    int lnid;
    int c = d_RN_ARRAY_COUNTER[list_ind];
    if (c == 0){
        return;
    }
    else{
        --c;
    }

    // lnid=d_ACCEPTORS[tmp_ind];



    double dr_sq = 0.0;
    double dr0 = 0.0;
    double dr_arr[3];
    double delr = r_n + 1.0;
    double dU = 0.0;
    int nid;
    int r;


    dr_sq = 0.0;
    dr0 = 0.0;
    dU = 0.0;


    l_state = d_states[ind];
    r = (int)round((curand_uniform(&l_state) * c));
    d_states[ind] = l_state;
    lnid = d_RN_ARRAY[list_ind * sticker_density * nncells +r];


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



    if (atomicCAS(&d_BONDS.get()[lnid * 2], 0, -1) == 0){ //lock the particle to bond with

        curandState l_state;
        l_state = d_states[ind];
        float rnd = curand_uniform(&l_state);
        d_states[ind] = l_state;

        if (rnd < exp(e_bond/2.0 - dU))
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
    int r_n,
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



    if (rnd > 0.05){return;} // only run on 1% of particles to avoid excessive fluctuations


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


    if (rnd < exp(-e_bond/2.0 + dU))
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



__global__ void d_update_grid(
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
    const int sticker_density,
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
    if (insrt_pos < sticker_density){
        d_MASTER_GRID[cell_id * sticker_density + insrt_pos] = list_ind;
    }
    else{
        ++d_LOW_DENS_FLAG[list_ind];
    }
}


__global__ void d_update_neighbours(
    const float *x,
    const float *Lh,
    const float *L,
    thrust::device_ptr<int> d_BONDS,
    thrust::device_ptr<int> d_MASTER_GRID_counter,
    thrust::device_ptr<int> d_MASTER_GRID,
    thrust::device_ptr<int> d_Nxx,
    thrust::device_ptr<float> d_Lg,
    thrust::device_ptr<int> d_RN_ARRAY,
    thrust::device_ptr<int> d_RN_ARRAY_COUNTER,
    thrust::device_ptr<int> d_DONORS,
    const int nncells,
    const int n_donors,
    const float r_n,
    const int sticker_density,
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

    int nxi, nyi, nzi, nid, counter, lnid;
    counter = 0;

    int ngs[27];

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

            float dist = 0.0f;                
            float dr_2 = 0.0f;
            float dr0;

            lnid = d_MASTER_GRID[ngs[i] * sticker_density + j];
    
            nid = d_index[lnid];

            for (int j = 0; j < D; j++){
                dr0 = my_x[j] - x[nid * D + j];

                if (dr0 >  Lh[j]){dr_arr[j] = -1.0f * (L[j] - dr0);}
                else if (dr0 < -1.0f * Lh[j]){dr_arr[j] = (L[j] + dr0);}
                else{dr_arr[j] = dr0;}

                dr_2 += dr_arr[j] * dr_arr[j];
            }
            if (dr_2 > 1.0E-5f) {
                dist = sqrt(dr_2);
            }
            else{
                dist = 0.0f;
            }
            if (dist <= r_n && d_BONDS[2*lnid] ==0){
                int insrt_pos = atomicAdd(&d_RN_ARRAY_COUNTER.get()[list_ind], 1);
                if (insrt_pos < sticker_density * nncells){
                    d_RN_ARRAY[list_ind * sticker_density*nncells + insrt_pos] = lnid;
                }
            }
        }
    } 
}