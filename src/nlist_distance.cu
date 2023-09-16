// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "nlist_distance.h"
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

NListDistance::~NListDistance() { return; }

NListDistance::NListDistance(istringstream &iss) : NList(iss)

{   
    if (style == "distance"){

        id = total_num_nlists++;
        trig_id = total_num_triggers++;

        int tmp;
        xyz = 1;

        readRequiredParameter(iss, r_n);
        readRequiredParameter(iss, r_skin);
        readRequiredParameter(iss, ad_hoc_density);
        readRequiredParameter(iss, nlist_freq);
        readRequiredParameter(iss, file_name);

        delta_r = r_skin - r_n;
        if (delta_r <= 0){die("delta_r cannot be negative or 0!");}
        dr_Triggers.push_back(delta_r);

        std::cout << "Group name: " << group_name << ", id: " << id << endl;
        std::cout << "Style: " << style << endl;
        std::cout << "r_n: " << r_n << ", r_skin: " << r_skin <<  ", delta_r: " << delta_r << ", nlist_freq: " << nlist_freq << endl;

        for (int i = 0; i < Dim; i++)
        {
            tmp = floor(L[i]/r_skin);
            Nxx.push_back(tmp);
            xyz *= int(tmp);
        }

        for (int i = 0; i < Dim; i++)
        {
            Lg.push_back(L[i] / float(Nxx[i]));
        }
        for (int i = 0; i < Dim; i++)
        {
            std::cout << "Nxx[" << i << "]: " << Nxx[i] << " |L:" << L[i] << " |dL: " << Lg[i] << endl;
        }
    }

    else if(style == "grid"){

        id = total_num_nlists++;
        int tmp;
        xyz = 1;
        std::cout << "Group name: " << group_name << ", id: " << id << endl;
        std::cout << "Style: " << style << endl;
        for (int i = 0; i < Dim; i++)
        {
            iss >> tmp;
            Nxx.push_back(tmp);
            xyz *= int(tmp);
        }

        for (int i = 0; i < Dim; i++)
        {
            Lg.push_back(L[i] / float(Nxx[i]));
        }
        for (int i = 0; i < Dim; i++)
        {
            std::cout << "Nx[" << i << "]: " << Nxx[i] << " | " << L[i] << " | " << Lg[i] << endl;
        }

        readRequiredParameter(iss, ad_hoc_density);
        readRequiredParameter(iss, nlist_freq);
        readRequiredParameter(iss, file_name);
        r_skin = -1.0;
    }

    std::cout << "File name: " << file_name << endl;

    if (Dim == 2){Nxx.push_back(1); Lg[2] = 1.0;}

    nncells = int(pow(3, Dim));          

    MASTER_GRID.resize(xyz * ad_hoc_density);                 
    MASTER_GRID_counter.resize(xyz);
    RN_ARRAY.resize(group->nsites * ad_hoc_density * nncells);
    RN_ARRAY_COUNTER.resize(group->nsites);
    d_LOW_DENS_FLAG.resize(group->nsites);

    for (int j = 0; j < xyz * ad_hoc_density; ++j)
    {
        MASTER_GRID[j] = -1;
    }
    for (int j = 0; j < xyz; ++j)
    {
        MASTER_GRID_counter[j] = 0;
    }

    for (int j = 0; j < group->nsites * ad_hoc_density * nncells; ++j){ RN_ARRAY[j] = -1;}
    for (int j = 0; j < group->nsites; ++j){ RN_ARRAY_COUNTER[j] = 0;}

    d_Nxx = Nxx;
    d_Lg = Lg;

    d_MASTER_GRID = MASTER_GRID;
    d_MASTER_GRID_counter = MASTER_GRID_counter;
    d_RN_ARRAY = RN_ARRAY;
    d_RN_ARRAY_COUNTER = RN_ARRAY_COUNTER;

    std::cout << "Distance parameters || xyz: " << xyz << ", ad_hoc_density: " << ad_hoc_density << ", Dim: " << Dim << ", n_cells: " << nncells << endl;
    std::cout << "Size: " << d_RN_ARRAY.size() << endl;
    KillingMeSoftly();
}

void NListDistance::MakeNList()
{   


    int sum = thrust::reduce(d_LOW_DENS_FLAG.begin(), d_LOW_DENS_FLAG.end(), 0, thrust::plus<int>());
    LOW_DENS_FLAG = float(sum)/float(MASTER_GRID_counter.size());
    if (LOW_DENS_FLAG > 0){
        
        cout << "Input density was: " << ad_hoc_density <<" but at least "<< ad_hoc_density + LOW_DENS_FLAG <<" is required"<<endl;
        ad_hoc_density += ceil(LOW_DENS_FLAG*1.5);
        cout << "Increasing the density to " <<  ad_hoc_density <<  " at step " << step << endl;

        d_MASTER_GRID.resize(xyz * ad_hoc_density);                 
        d_RN_ARRAY.resize(group->nsites * ad_hoc_density * nncells);

        MASTER_GRID.resize(xyz * ad_hoc_density);                 
        RN_ARRAY.resize(group->nsites * ad_hoc_density * nncells);

        LOW_DENS_FLAG = 0;
    }

    thrust::fill(d_MASTER_GRID.begin(), d_MASTER_GRID.end(), 0);
    thrust::fill(d_MASTER_GRID_counter.begin(), d_MASTER_GRID_counter.end(), 0);
    thrust::fill(d_RN_ARRAY.begin(), d_RN_ARRAY.end(), 0);
    thrust::fill(d_RN_ARRAY_COUNTER.begin(), d_RN_ARRAY_COUNTER.end(), 0);
    thrust::fill(d_LOW_DENS_FLAG.begin(), d_LOW_DENS_FLAG.end(), 0);

    //cudaDeviceSynchronize();

    // for (int i = 0; i < MASTER_GRID.size(); ++i){
    //     MASTER_GRID[i] = 0;
    // }
    // d_MASTER_GRID = MASTER_GRID;

    // for (int i = 0; i < MASTER_GRID_counter.size(); ++i){
    //     MASTER_GRID_counter[i] = 0;
    // }
    // d_MASTER_GRID_counter = MASTER_GRID_counter;

    // for (int i = 0; i < RN_ARRAY.size(); ++i){
    //     RN_ARRAY[i] = 0;
    // }
    // d_RN_ARRAY = RN_ARRAY;    

    // for (int i = 0; i < RN_ARRAY_COUNTER.size(); ++i){
    //     RN_ARRAY_COUNTER[i] = 0;
    // }
    // d_RN_ARRAY_COUNTER = RN_ARRAY_COUNTER;

    // thrust::fill(d_MASTER_GRID.begin(), d_MASTER_GRID.end(), 0);
    // thrust::fill(d_MASTER_GRID_counter.begin(), d_MASTER_GRID_counter.end(), 0);
    // thrust::fill(d_RN_ARRAY.begin(), d_RN_ARRAY.end(), 0);
    // thrust::fill(d_RN_ARRAY_COUNTER.begin(), d_RN_ARRAY_COUNTER.end(), 0);
    // thrust::fill(d_LOW_DENS_FLAG.begin(), d_LOW_DENS_FLAG.end(), 0);
    //cudaDeviceSynchronize();

    // d_nlist_distance_update_grid<<<GRID, BLOCK>>>(d_x,
    //     d_MASTER_GRID_counter.data(), d_MASTER_GRID.data(),
    //     d_Nxx.data(), d_Lg.data(),
    //     ad_hoc_density,
    //     group->d_index, group->nsites, Dim, step, d_LOW_DENS_FLAG.data());

    //cudaDeviceSynchronize();

    // d_nlist_distance_update_nlist<<<GRID, BLOCK>>>(d_x, d_Lh, d_L,
    //     d_MASTER_GRID_counter.data(), d_MASTER_GRID.data(),
    //     d_Nxx.data(), d_Lg.data(),
    //     d_RN_ARRAY.data(), d_RN_ARRAY_COUNTER.data(),
    //     ad_hoc_density,
    //     group->d_index, group->nsites, Dim,
    //     nncells, r_skin, step);


    
    d_nlist_distance_update<<<GRID, BLOCK>>>(d_x, d_Lh, d_L,
        d_MASTER_GRID_counter.data(), d_MASTER_GRID.data(),
        d_Nxx.data(), d_Lg.data(),
        d_RN_ARRAY.data(), d_RN_ARRAY_COUNTER.data(),
        d_LOW_DENS_FLAG.data(),
        step,nncells,r_skin, ad_hoc_density,
        group->d_index, group->nsites, Dim);

        
    //cudaDeviceSynchronize();

    // WriteNList();

}


__global__ void d_nlist_distance_update(
    const float *x, // [ns*Dim], particle positions
    const float *Lh,
    const float *L,
    thrust::device_ptr<int> d_MASTER_GRID_counter,
    thrust::device_ptr<int> d_MASTER_GRID,
    thrust::device_ptr<int> d_Nxx,
    thrust::device_ptr<float> d_Lg,
    thrust::device_ptr<int> d_RN_ARRAY,
    thrust::device_ptr<int> d_RN_ARRAY_COUNTER,
    thrust::device_ptr<int> d_LOW_DENS_FLAG,
    int step,
    const int nncells,
    const float r_skin,
    const int ad_hoc_density,
    const int *site_list, // List of sites in the group
    const int ns,         // Number of sites in the list
    const int D)
{

    int list_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (list_ind >= ns)
        return;
    int ind = site_list[list_ind];

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
        // printf("Warning! Grid density too low at step %d!\n", step);
        ++d_LOW_DENS_FLAG[list_ind];
    }

    __syncthreads();

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

    // if (r_skin < 0){
    //     for (int i = 0; i < nncells; ++i){
    //         int m = d_MASTER_GRID_counter[ngs[i]];
    //         for (int j = 0; j < d_MASTER_GRID_counter[ngs[i]]; j++){
    //             lnid = d_MASTER_GRID[ngs[i] * ad_hoc_density + j];
    //             if (lnid != list_ind){
    //                 int insrt_pos = atomicAdd(&d_RN_ARRAY_COUNTER.get()[list_ind], 1);
    //                 d_RN_ARRAY[list_ind * ad_hoc_density*nncells + insrt_pos] = lnid;
    //             }
    //         }
    //     } 
    //     delete[] ngs;       
    // }

    // else{

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
                if (lnid != list_ind){
                    nid = site_list[lnid];

                    for (int j = 0; j < D; j++){
                        dr0 = my_x[j] - x[nid * D + j];

                        if (dr0 >  Lh[j]){dr_arr[j] = -1.0 * (L[j] - dr0);} // pbc
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
                        d_RN_ARRAY[list_ind * ad_hoc_density*nncells + insrt_pos] = lnid;
                    }
                }
            }
        } 
        delete[] ngs;       
    // }
}







__global__ void d_nlist_distance_update_grid(
    const float *x, // [ns*Dim], particle positions
    thrust::device_ptr<int> d_MASTER_GRID_counter,
    thrust::device_ptr<int> d_MASTER_GRID,
    thrust::device_ptr<int> d_Nxx,
    thrust::device_ptr<float> d_Lg,
    const int ad_hoc_density,
    const int *site_list, // List of sites in the group
    const int ns,         // Number of sites in the list
    const int D,
    int step, thrust::device_ptr<int> d_LOW_DENS_FLAG)
{

    int list_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (list_ind >= ns)
        return;
    int ind = site_list[list_ind];

    // calculate flattened position

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
        // printf("Warning! Grid density too low at step %d!\n", step);
        ++d_LOW_DENS_FLAG[list_ind];
    }


}


__global__ void d_nlist_distance_update_nlist(
    const float *x, // [ns*Dim], particle positions
    const float *Lh,
    const float *L,
    thrust::device_ptr<int> d_MASTER_GRID_counter,
    thrust::device_ptr<int> d_MASTER_GRID,
    thrust::device_ptr<int> d_Nxx,
    thrust::device_ptr<float> d_Lg,
    thrust::device_ptr<int> d_RN_ARRAY,
    thrust::device_ptr<int> d_RN_ARRAY_COUNTER,
    const int ad_hoc_density,
    const int *site_list, // List of sites in the group
    const int ns,         // Number of sites in the list
    const int D,
    const int nncells,
    const float r_skin,
    const int step)
{

    int list_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (list_ind >= ns)
        return;
    int ind = site_list[list_ind];

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

    if (r_skin < 0){
        for (int i = 0; i < nncells; ++i){
            int m = d_MASTER_GRID_counter[ngs[i]];
            for (int j = 0; j < d_MASTER_GRID_counter[ngs[i]]; j++){
                lnid = d_MASTER_GRID[ngs[i] * ad_hoc_density + j];
                if (lnid != list_ind){
                    int insrt_pos = atomicAdd(&d_RN_ARRAY_COUNTER.get()[list_ind], 1);
                    d_RN_ARRAY[list_ind * ad_hoc_density*nncells + insrt_pos] = lnid;
                }
            }
        } 
        delete[] ngs;       
    }

    else{

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
                if (lnid != list_ind){
                    nid = site_list[lnid];

                    for (int j = 0; j < D; j++){
                        dr0 = my_x[j] - x[nid * D + j];

                        if (dr0 >  Lh[j]){dr_arr[j] = -1.0 * (L[j] - dr0);} // pbc
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
                        d_RN_ARRAY[list_ind * ad_hoc_density*nncells + insrt_pos] = lnid;
                    }
                }
            }
        } 
        delete[] ngs;       
    }
}


void NListDistance::WriteNList(void)
{
    const char* fname = (file_name + "_grid").c_str();
    if (step == 0){remove(fname);}

    MASTER_GRID = d_MASTER_GRID;
    MASTER_GRID_counter = d_MASTER_GRID_counter;
    RN_ARRAY = d_RN_ARRAY;
    RN_ARRAY_COUNTER = d_RN_ARRAY_COUNTER;

    ofstream nlist_file;
    nlist_file.open(file_name + "_grid", ios::out | ios::app);
    nlist_file << "TIMESTEP: " << step << endl;
    for (int j = 0; j < xyz; ++j){
        nlist_file << j << "|" << MASTER_GRID_counter[j] << ": ";
        for (int i = 0; i < ad_hoc_density; i++){
            if (MASTER_GRID[j*ad_hoc_density + i] != -1)
            nlist_file << group->index[MASTER_GRID[j*ad_hoc_density + i]] << " ";
            else nlist_file << "* ";
        }
        nlist_file << endl;
    }

    const char* pfname = (file_name + "_pp").c_str();
    if (step == 0){remove(pfname);}

    ofstream pnlist_file;
    pnlist_file.open(file_name + "_pp", ios::out | ios::app);
    pnlist_file << "TIMESTEP: " << step << endl;
    for (int j = 0; j < group->nsites; ++j){
        pnlist_file << group->index[j] << "|" << RN_ARRAY_COUNTER[j] << ": ";
        for (int i = 0; i < nncells * ad_hoc_density; i++){
            if (RN_ARRAY[j * nncells * ad_hoc_density + i] != -1)
            pnlist_file << group->index[RN_ARRAY[j * nncells * ad_hoc_density + i]] <<" ";
            else pnlist_file <<"* ";
        }
        pnlist_file << endl;
    }
}