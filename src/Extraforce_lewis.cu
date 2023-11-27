// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "Extraforce_lewis.h"
#include "Extraforce_dynamic.h"
#include "potential_charges.h"
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

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
# include <algorithm>

#define EPSILON 1.0e-10

using namespace std;


Lewis::~Lewis() { return; }

Lewis::Lewis(istringstream &iss) : ExtraForce(iss)
{


    DynamicBonds.push_back(this);
    thrust::default_random_engine g(time(0));


    cout << "Lewis bonds active!" << endl;

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
        std::cout << "Donors >= Acceptors" << std::endl;
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

    // for (auto element: DONORS){
    //     std::cout << element << ", ";
    // }




    readRequiredParameter(iss, k_spring);
    cout << "k_spring: " << k_spring << endl;
    readRequiredParameter(iss, e_bond);
    cout << "e_bond: " << e_bond << endl;
    readRequiredParameter(iss, qind);
    cout << "induced charge: " << qind << endl;
    readRequiredParameter(iss, r0);
    cout << "r0: " << r0 << endl;
    readRequiredParameter(iss, r_n);
    cout << "r_n: " << r_n << endl;
    readRequiredParameter(iss, sticker_density);
    cout << "sticker_density: " << sticker_density << endl;
    readRequiredParameter(iss, active_fraction);
    cout << "Active fraction: " << active_fraction << endl;
    readRequiredParameter(iss, bond_freq);
    cout << "bond_freq: " << bond_freq << endl;
    readRequiredParameter(iss, bond_log_freq);
    cout << "bond_log_freq: " << bond_log_freq << endl;
    readRequiredParameter(iss, file_name);
    cout << "output_file: " << file_name << endl;
    readRequiredParameter(iss, offset);
    cout << "offset: " << offset << endl;

    std::vector<string> extra_args;
    std::string temp_str;

    while(iss >> temp_str){
        extra_args.push_back(temp_str);
    }

    int i = 0;
    int RESUME_BONDS = 0;
    RAMP_FLAG = 0;
    std::string bonds_resume_file;

    

    while (i < extra_args.size()){
        if (extra_args[i] == "ramp"){
            ++i;
            std::cout << "Energy ramp activated!" << std::endl;
    
            e_bond_final = std::stof(extra_args[i++]);
            ramp_reps = std::stoi(extra_args[i++]);
            ramp_t_end = std::stoi(extra_args[i++]);
    
            ramp_interval = ceil(float(ramp_t_end)/float(ramp_reps));
            ramp_counter = 0;
            RAMP_FLAG = 1;
            delta_e_bond = (e_bond_final - e_bond)/float(ramp_reps);
            std::cout << "Final energy: "<< e_bond_final <<", in " << ramp_reps <<" intervals of " << ramp_interval << " time steps" << std::endl;
        }

        else if (extra_args[i] == "resume"){
            ++i;
            RESUME_BONDS = 1;
            bonds_resume_file = extra_args[i++];
        }

        else{
            ++i;
        }


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
    d_FREE_ACCEPTORS.resize(n_donors);

    BONDED.resize( n_donors);
    FREE.resize( n_donors);
    FREE_ACCEPTORS.resize(n_donors);


    for (int i = 0; i < n_donors; ++i){
        FREE[i] = d_DONORS[i];
        BONDED[i] = -1;
    }

    // for (int i = 0; i < n_acceptors; ++i){
    //     FREE_ACCEPTORS[i] = d_ACCEPTORS[i];
    // }


    n_bonded = 0;
    n_free_donors = n_donors;
    n_free_acceptors = n_acceptors;

    d_FREE = FREE;
    d_BONDED = BONDED;
    d_FREE_ACCEPTORS = FREE_ACCEPTORS;

    GRID = (int)ceil(((float)max(n_donors,n_acceptors))/float(threads));

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
    RN_ARRAY.resize(group->nsites * sticker_density * nncells);
    d_RN_ARRAY_COUNTER.resize(group->nsites);

    d_RN_DISTANCE.resize(group->nsites * sticker_density * nncells);
    RN_DISTANCE.resize(group->nsites * sticker_density * nncells);

    ACCEPTOR_LOCKS.resize(group->nsites);



    d_LOW_DENS_FLAG.resize(group->nsites);

    thrust::fill(d_MASTER_GRID.begin(),d_MASTER_GRID.end(),-1);
    thrust::fill(d_MASTER_GRID_counter.begin(),d_MASTER_GRID_counter.end(),0);

    thrust::fill(d_RN_ARRAY.begin(),d_RN_ARRAY.end(),-1);
    thrust::fill(d_RN_ARRAY_COUNTER.begin(),d_RN_ARRAY_COUNTER.end(),0);

    thrust::fill(d_LOW_DENS_FLAG.begin(), d_LOW_DENS_FLAG.end(), 0);

    std::cout << "Total cells: " << xyz << ", sticker_density: " << sticker_density << ", Dim: " << Dim << ", n_cells: " << nncells << endl;

    if (RESUME_BONDS ==1){
        std::cout << "Resume file for bonds read" << std::endl;


        std::ifstream infile("resume_bonds");
        std::string line;

        std::getline(infile, line);
        std::istringstream iss(line);
        int ts;
        std::string TS;
        iss >> TS >> ts;
        std::cout << "Resuming at TIMESTEP: " << ts << std::endl;

        while (std::getline(infile, line))
        {
            std::istringstream iss(line);
            int d,a;
            if (!(iss >> d >> a)) { break; } // error
        
            --d;
            --a;

            std::cout << d << " " << a << std::endl;

            d = std::distance(group->index.begin(), std::find(group->index.begin(), group->index.end(), d));
            a = std::distance(group->index.begin(), std::find(group->index.begin(), group->index.end(), a));
            std::cout << d << " " << a << std::endl << std::endl;

            BONDS[d * 2] = 1;
            BONDS[a * 2] = 1;

            BONDS[d * 2 + 1] = a;
            BONDS[a * 2 + 1] = d;
        }

        d_BONDS = BONDS;
        UpdateBonders();
    }




}


void Lewis::IncreaseCapacity(){

        // increase the capacity of the density array if it is not enough to hold all the particles

        int sum = thrust::count(d_LOW_DENS_FLAG.begin(), d_LOW_DENS_FLAG.end(), 1);
        float ldc = float(sum);

        cout << "Input sticker density was: " << sticker_density <<" but at least "<< sticker_density + ldc <<" is required"<<endl;
        sticker_density += int(ceil(ldc * 5));
        cout << "Increasing sticker density to " <<  sticker_density <<  " at step " << step << endl;

        d_MASTER_GRID.resize(xyz * sticker_density);                 
    
        d_RN_ARRAY.resize(group->nsites * sticker_density * nncells);
        RN_ARRAY.resize(group->nsites * sticker_density * nncells);
    
        d_RN_DISTANCE.resize(group->nsites * sticker_density * nncells);
        RN_DISTANCE.resize(group->nsites * sticker_density * nncells);

        thrust::fill(d_MASTER_GRID.begin(),d_MASTER_GRID.end(),-1);
        thrust::fill(d_MASTER_GRID_counter.begin(),d_MASTER_GRID_counter.end(),0);
        thrust::fill(d_LOW_DENS_FLAG.begin(), d_LOW_DENS_FLAG.end(), 0);

        d_update_grid<<<GRID, threads>>>(d_x, d_Lh, d_L,
            d_MASTER_GRID_counter.data(), d_MASTER_GRID.data(),
            d_Nxx.data(), d_Lg.data(),
            d_LOW_DENS_FLAG.data(),
            d_ACCEPTORS.data(),
            nncells, n_acceptors, sticker_density,
            group->d_index.data(), group->nsites, Dim);


        sum = thrust::reduce(d_LOW_DENS_FLAG.begin(), d_LOW_DENS_FLAG.end(), 0, thrust::plus<int>());
        LOW_DENS_FLAG = float(sum)/float(d_MASTER_GRID_counter.size());

        if (LOW_DENS_FLAG > 0){
            die("Error Resizing Arrays!");
        } 

    
}


void Lewis::UpdateBonders(){

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
         n_free_acceptors++;
        }
    }

    // for (auto element : FREE_ACCEPTORS){
    //     std::cout << element << " ";
    // }
    // std::cout << std::endl << std::endl;

    d_BONDED = BONDED;
    d_FREE = FREE;


}

void Lewis::UpdateNList(){

        // Updates n-list for the donors
        thrust::fill(d_RN_ARRAY.begin(),d_RN_ARRAY.end(),-1);
        thrust::fill(d_RN_ARRAY_COUNTER.begin(),d_RN_ARRAY_COUNTER.end(),0);
        thrust::fill(RN_DISTANCE.begin(),RN_DISTANCE.end(),1000.0);
        d_RN_DISTANCE = RN_DISTANCE;

        d_update_neighbours<<<GRID, threads>>>(d_x, d_Lh, d_L,
            d_BONDS.data(),
            d_MASTER_GRID_counter.data(), d_MASTER_GRID.data(),
            d_Nxx.data(), d_Lg.data(),
            d_RN_ARRAY.data(), d_RN_ARRAY_COUNTER.data(),
            d_RN_DISTANCE.data(),
            d_DONORS.data(),
            nncells, n_donors, r_n, sticker_density,
            group->d_index.data(), group->nsites, Dim);


        // Updates the distribution of acceptors on the grid
        
        RN_ARRAY = d_RN_ARRAY;
        RN_ARRAY_COUNTER = d_RN_ARRAY_COUNTER; 
        RN_DISTANCE = d_RN_DISTANCE;


        // sort the neighboour by distance
        // shuffle the donors for stochastic results

        thrust::fill(ACCEPTOR_LOCKS.begin(),ACCEPTOR_LOCKS.end(),0);
        thrust::fill(FREE_ACCEPTORS.begin(),FREE_ACCEPTORS.end(),-1);
        thrust::shuffle(FREE.begin(), FREE.begin() + n_free_donors, g);

        d_FREE = FREE;


        for (int i = 0; i < n_free_donors; ++i){


            int list_ind = FREE[i];

            int c = RN_ARRAY_COUNTER[list_ind];

            if (c!=0){
    


                int offset = list_ind * sticker_density * nncells;

                // thrust::sort_by_key(thrust::host, keys, keys + N, values);

            //     if (i == 2 || i == 10){
            //     for (int j = 0; j < c; ++j){
            //         std::cout << RN_ARRAY[offset + j] << " " << RN_DISTANCE[offset+j] << ", ";
            //     }
            //     std::cout << std::endl << std::endl;
            // }


                thrust::sort_by_key(thrust::host, RN_DISTANCE.begin() + offset, RN_DISTANCE.begin() + offset + c, RN_ARRAY.begin() + offset);

                // if (i == 2 || i == 10){
                //     for (int j = 0; j < c; ++j){
                //         std::cout << RN_ARRAY[offset + j] << " " << RN_DISTANCE[offset+j] << ", ";
                //     }
                //     std::cout << std::endl << std::endl;
                // }
                


                int lnid;
                // if (i==2 || i == 10)
                // std::cout << lnid << ", ";

                int count = 0;
                while (count < c){
                    lnid = RN_ARRAY[offset  + count ];
                    if (ACCEPTOR_LOCKS[lnid] == 0){
                        ACCEPTOR_LOCKS[lnid] = 1;
                        FREE_ACCEPTORS[i] = lnid;
                        break;
                    }
                        ++count;
                }


                    // lnid = RN_ARRAY[offset];
                    // if (ACCEPTOR_LOCKS[lnid] == 0){
                    //     ACCEPTOR_LOCKS[lnid] = 1;
                    //     FREE_ACCEPTORS[i] = lnid;
                    // }

            }
        }
        // std::cout <<  std::endl<<std::endl;
        d_FREE_ACCEPTORS = FREE_ACCEPTORS;

        // d_RN_ARRAY = RN_ARRAY; // ordered neighbour list

}

void Lewis::AddExtraForce()
{   


    // if (RAMP_FLAG == 1 && ramp_counter < ramp_reps && step % ramp_interval == 0 && step > 0){
    //     e_bond += delta_e_bond;
    //     std::cout << "At step: " << step <<" increased e_bond to: " << e_bond << std::endl;
    //     ++ramp_counter;
    // }



    if (step % bond_freq == 0 && step > 0){


        thrust::fill(d_MASTER_GRID.begin(),d_MASTER_GRID.end(),-1);
        thrust::fill(d_MASTER_GRID_counter.begin(),d_MASTER_GRID_counter.end(),0);
        thrust::fill(d_LOW_DENS_FLAG.begin(), d_LOW_DENS_FLAG.end(), 0);

        d_update_grid<<<GRID, threads>>>(d_x, d_Lh, d_L,
            d_MASTER_GRID_counter.data(), d_MASTER_GRID.data(),
            d_Nxx.data(), d_Lg.data(),
            d_LOW_DENS_FLAG.data(),
            d_ACCEPTORS.data(),
            nncells, n_acceptors, sticker_density,
            group->d_index.data(), group->nsites, Dim);
        int sum = thrust::reduce(d_LOW_DENS_FLAG.begin(), d_LOW_DENS_FLAG.end(), 0, thrust::plus<int>());
        LOW_DENS_FLAG = float(sum)/float(d_MASTER_GRID_counter.size());

        if (LOW_DENS_FLAG > 0){
            IncreaseCapacity();
        }

        for (int i = 0; i < 2; i++){
            

            if (n_free_donors > 0){

            UpdateNList();

            // Electrostatics - Begin

            prepareDensityFields();
            MasterCharge->CalcCharges();
            MasterCharge->CalcEnergy();

            // Electrostatics - End


            int BLOCKS = (int)ceil((n_free_donors * 2.0 * (float)n_acceptors/(n_donors + n_acceptors) * active_fraction)/(float)threads);

            d_make_bonds_lewis<<<1, 1>>>(d_x,d_f,
                d_BONDS.data(),
                d_FREE_ACCEPTORS.data(),
                d_FREE.data(),
                d_RN_ARRAY.data(),d_RN_ARRAY_COUNTER.data(),
                d_VirArr.data(), n_free_donors,
                n_donors,n_acceptors,
                sticker_density,nncells,
                group->d_index.data(), group->nsites, d_states,
                k_spring, e_bond, r0, r_n,
                active_fraction,
                d_L, d_Lh, Dim,
                qind,
                d_charges,
                grid_per_partic,
                d_electrostatic_potential,
                d_grid_inds,
                d_grid_W);

                
            check_cudaError("Extraforce_Lewis.cu: d_make_bonds_lewis");

            // Electrostatics - Begin

            cudaMemcpy(charges, d_charges, ns * sizeof(float), cudaMemcpyDeviceToHost);
            check_cudaError("Extraforce_lewis.cu: cudaMemcpy(charges, d_charges, ns * sizeof(float), cudaMemcpyDeviceToHost);");

            // Electrostatics - End

            } // if n_free_donors > 0



            thrust::shuffle(BONDED.begin(),BONDED.begin() + n_bonded,g);
            d_BONDED = BONDED;


            if (n_bonded > 0){
                
                prepareDensityFields();
                MasterCharge->CalcCharges();
                MasterCharge->CalcEnergy();

                int BLOCKS = (int)ceil((n_bonded *float(n_bonded)/(n_donors + n_acceptors)*2.0 * active_fraction)/(float)threads);


                d_break_bonds_lewis<<<1, 1>>>(d_x,
                    d_BONDS.data(),
                    d_BONDED.data(),n_bonded,
                    n_donors,n_acceptors,
                    r_n,
                    group->d_index.data(), group->nsites, d_states,
                    k_spring, e_bond, r0,
                    active_fraction,
                    d_L, d_Lh, Dim,
                    qind,
                    d_charges,
                    grid_per_partic,
                    d_electrostatic_potential,
                    d_grid_inds,
                    d_grid_W);

                
                    check_cudaError("Extraforce_Lewis.cu: d_break_bonds_lewis");

                    // Electrostatics - Begin

                    cudaMemcpy(charges, d_charges, ns * sizeof(float), cudaMemcpyDeviceToHost);
                    check_cudaError("Extraforce_lewis.cu: cudaMemcpy(charges, d_charges, ns * sizeof(float), cudaMemcpyDeviceToHost);");

                    // Electrostatics - End

                } // if n_free_donors > 0

            UpdateBonders();
            prepareDensityFields();
            MasterCharge->CalcCharges();
            MasterCharge->CalcEnergy();

        }// loop

    } // end if (step % lewis_bond_freq == 0 && step >= bond_freq)

    if(step >= bond_freq){
        d_update_forces<<<GRID, threads>>>(d_x, d_f, d_L, d_Lh,
            k_spring, e_bond, r0,
            d_BONDS.data(), d_BONDED.data(), d_VirArr.data(), n_bonded,
            group->d_index.data(), group->nsites, Dim);
    }

    // if (step == 0){
    //     const char* fname = file_name.c_str();
    //     remove(fname);
    // }

    if (step % bond_log_freq == 0 && step >= offset)
    {
        Lewis::WriteBonds();
    }

    if (step % traj_freq == 0 && traj_freq > 0)
    {
        Lewis::write_resume_files();
    }
}


/*
Updates forces acting on particles due to Lewis bonds
*/

__global__ void d_make_bonds_lewis(
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
    float active_fraction,
    float *L,
    float *Lh,
    int D,
    float qind,
    float* d_charges,
    int grid_per_partic,
    float* d_electrostatic_potential,
    int* d_grid_inds,
    float* d_grid_W)

{

    int tmp_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (tmp_ind >= n_free_donors * 2.0 * (float)n_acceptors/(n_donors + n_acceptors) * active_fraction)
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
  
    // if (rnd > active_fraction){return;} // only run on 1% of particles to avoid excessive fluctuations

    // l_state = d_states[ind];
    // rnd = curand_uniform(&l_state);
    // d_states[ind] = l_state;


    int lnid;
    int c = d_RN_ARRAY_COUNTER[list_ind];
    if (c == 0){
        return;
    }
    else{
        --c;
    }

    lnid=d_ACCEPTORS[tmp_ind];

    // printf(" -- %d -- ",lnid);
    if (lnid == -1){
        return;
    }



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


    // l_state = d_states[ind];
    // r = (int)round((curand_uniform(&l_state) * c));
    // d_states[ind] = l_state;
    // lnid = d_RN_ARRAY[list_ind * sticker_density * nncells +r];


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


    for (int grid_ct=0; grid_ct < grid_per_partic; ++grid_ct){
        dU += qind * d_electrostatic_potential[d_grid_inds[ind * grid_per_partic + grid_ct]] * d_grid_W[ind * grid_per_partic + grid_ct];
        dU -= qind * d_electrostatic_potential[d_grid_inds[nid * grid_per_partic + grid_ct]] * d_grid_W[nid * grid_per_partic + grid_ct];
    }



    if (atomicCAS(&d_BONDS.get()[lnid * 2], 0, -1) == 0){ //lock the particle to bond with

        curandState l_state;
        l_state = d_states[ind];
        float rnd = curand_uniform(&l_state);
        d_states[ind] = l_state;


            if (rnd < 1.0/(1+exp(dU)))
            // if (rnd < exp(e_bond/2.0))
            {
                atomicExch(&d_BONDS.get()[list_ind * 2], 1);
                atomicExch(&d_BONDS.get()[lnid * 2], 1);

                atomicExch(&d_BONDS.get()[list_ind * 2 + 1], lnid);
                atomicExch(&d_BONDS.get()[lnid * 2 + 1], list_ind);

                d_charges[ind] += qind; 
                d_charges[nid] -= qind;

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



__global__ void d_break_bonds_lewis(
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
    float active_fraction,
    float *L,
    float *Lh,
    int D,
    float qind,
    float* d_charges,
    int grid_per_partic,
    float* d_electrostatic_potential,
    int* d_grid_inds,
    float* d_grid_W)


{
    int tmp_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (tmp_ind >= n_bonded *float(n_bonded)/(n_donors + n_acceptors)*2.0 * active_fraction)
        return;

    int list_ind = d_BONDED[tmp_ind];
    int ind = d_index[list_ind];


    curandState l_state;
    l_state = d_states[ind];
    float rnd = curand_uniform(&l_state);
    d_states[ind] = l_state;

    
    int n_free_acceptors = n_acceptors - n_bonded;
    int n_free_donors = n_donors-n_bonded;


    // if (rnd > active_fraction){return;} // only run on 1% of particles to avoid excessive fluctuations

    // l_state = d_states[ind];
    // rnd = curand_uniform(&l_state);
    // d_states[ind] = l_state;

    // if (rnd > float(n_acceptors)/float(n_donors)){return;} // only run on 1% of particles to avoid excessive fluctuations


    int lnid = d_BONDS[list_ind * 2 + 1];
    int nid = d_index[lnid];

    double dr_sq = 0.0;
    double dr0;
    double dr_arr[3];
    double delr;
    double dU;

    // l_state = d_states[ind];
    // rnd = curand_uniform(&l_state);
    // d_states[ind] = l_state;


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

    for (int grid_ct=0; grid_ct < grid_per_partic; ++grid_ct){

        dU -= qind * d_electrostatic_potential[d_grid_inds[ind * grid_per_partic + grid_ct]] * d_grid_W[ind * grid_per_partic + grid_ct];
        dU += qind * d_electrostatic_potential[d_grid_inds[nid * grid_per_partic + grid_ct]] * d_grid_W[nid * grid_per_partic + grid_ct];

    }

    if (rnd < exp(-e_bond)/(1+exp(-dU)))
    // if (rnd < exp(-e_bond/2.0 + dU ))
    {
        atomicExch(&d_BONDS.get()[list_ind * 2], 0);
        atomicExch(&d_BONDS.get()[lnid * 2], 0);

        atomicExch(&d_BONDS.get()[list_ind * 2 + 1], -1);
        atomicExch(&d_BONDS.get()[lnid * 2 + 1], -1);

        d_charges[ind] -= qind; 
        d_charges[nid] += qind;
    }

}

void Lewis::WriteBonds(void)
{

    // int flag = 0;

    this->BONDS = d_BONDS;
    mbbond = d_mbbond;
    ofstream bond_file;
    bond_file.open(file_name, ios::out | ios::app);

    bond_file << global_step << " " << float(n_bonded)/min(n_donors,n_acceptors) << std::endl;//" " <<mbbond[0] <<" " << mbbond[1] << endl;

    bond_file.close();


    bond_file.open(file_name +"_log", ios::out | ios::app);

    bond_file << "TIMESTEP: " << global_step << " " << float(n_bonded)/min(n_donors,n_acceptors) << std::endl;//" " <<mbbond[0] <<" " << mbbond[1] << endl;
    for (int j = 0; j < group->nsites; ++j)
    {
        if (BONDS[2 * j + 1] != -1 && AD[j] == donor_tag)
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





void Lewis::write_resume_files(){

    FILE* otp;
    int i, j;

    std::string dump_name = "resume.lammsptrj";

    otp = fopen(dump_name.c_str(), "w");

    fprintf(otp, "ITEM: TIMESTEP\n%d\nITEM: NUMBER OF ATOMS\n%d\n", global_step, ns);
    fprintf(otp, "ITEM: BOX BOUNDS pp pp pp\n");
    fprintf(otp, "%f %f\n%f %f\n%f %f\n", 0.f, L[0],
        0.f, L[1],
        (Dim == 3 ? 0.f : 1.f), (Dim == 3 ? L[2] : 1.f));

    if ( Charges::do_charges )
        fprintf(otp, "ITEM: ATOMS id type mol x y z q\n");
    else
        fprintf(otp, "ITEM: ATOMS id type mol x y z\n");

    for (i = 0; i < ns; i++) {
        fprintf(otp, "%d %d %d  ", i + 1, tp[i] + 1, molecID[i] + 1);
        for (j = 0; j < Dim; j++)
            fprintf(otp, "%f ", x[i][j]);

        for (j = Dim; j < 3; j++)
            fprintf(otp, "%f", 0.f);

        if ( Charges::do_charges )
            fprintf(otp, " %f", charges[i]);

        fprintf(otp, "\n");
    }
    fclose(otp);


    this->BONDS = d_BONDS;
    ofstream bond_file;

    bond_file.open("resume_bonds", ios::out | ios::trunc);

    bond_file << "TIMESTEP: " << global_step << std::endl;

    for (int j = 0; j < group->nsites; ++j)
    {
        if (BONDS[2 * j + 1] != -1 && AD[j] == donor_tag)
        {
            bond_file << group->index[j] + 1 << " " << this->group->index[BONDS[2 * j + 1]] + 1 << endl;
        }
    }
    bond_file.close(); 

}