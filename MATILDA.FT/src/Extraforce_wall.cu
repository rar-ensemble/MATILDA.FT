// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include <curand_kernel.h>
#include <curand.h>
#include "Extraforce_wall.h"
#include "globals.h"
#include <thrust/copy.h>

using namespace std;

Wall::~Wall(){
return;
}
void Wall::UpdateVirial(void){return;};

Wall::Wall(istringstream& iss) : ExtraForce(iss){

    float tmp;
    w_size = 0;

    readRequiredParameter(iss, wall_style_str);
    cout << "Wall style is: " << wall_style_str << endl;

    if(wall_style_str == "hard"){wall_style = 0;}
    else if(wall_style_str == "rinv"){wall_style = 1;}
    else if(wall_style_str == "exp"){wall_style = 2;}

    while(iss >> tmp){
        d_wall.push_back(tmp); //dim l h
        ++w_size;
    }
    
}

void Wall::AddExtraForce() {

    if(wall_style == 0){
    d_wall_hard<<<group->GRID, group->BLOCK>>>(d_f, d_x, d_v, d_wall.data(),w_size,
        group->d_index.data(), group->nsites, Dim);
    }

    else if(wall_style == 1){
    d_wall_rinv<<<group->GRID, group->BLOCK>>>(d_f, d_x, d_v, d_wall.data(), w_size,
        group->d_index.data(), group->nsites, Dim);
    }

    else if(wall_style == 2){
    d_wall_exp<<<group->GRID, group->BLOCK>>>(d_f, d_x, d_v, d_wall.data(), w_size,
        group->d_index.data(), group->nsites, Dim);
    }
}

__global__ void d_wall_hard(
    float* f,             // [ns*Dim], particle forces
    float* x,            // [ns*Dim], particle positions
    float* v,
    thrust::device_ptr<float> d_w,
    const int w_size,
    thrust::device_ptr<int> d_index,  // List of sites in the group
    const int ns,           // Number of sites in the list
    const int D) {

    int list_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (list_ind >= ns)
        return;

    int ind = d_index[list_ind];

    float low, high;

    int nump = 3;
    int size = w_size/nump;

    for(int i = 0; i < size; ++i){
        int j = int(d_w[nump * i]); //dimension to act on; x:0, y:1; z:2
        low = d_w[nump * i + 1];
        high = d_w[nump * i + 2];
        int pi = ind * D + j;
        float my_x = x[pi];

        if (my_x < low) {
            x[pi] = low + (low - my_x);
            v[pi] = abs(v[pi]); // momentum points away from the wall
            }
        else if (my_x > high) {
            x[pi] = high - (my_x - high);
            v[pi] = -1.0 * abs(v[pi]);
            }
    }
}



__global__ void d_wall_rinv(
    float* f,             // [ns*Dim], particle forces
    float* x,            // [ns*Dim], particle positions
    float* v,
    thrust::device_ptr<float> d_w,
    const int w_size,
    thrust::device_ptr<int> d_index,  // List of sites in the group
    const int ns,           // Number of sites in the list
    const int D) {

    int list_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (list_ind >= ns)
        return;

    int ind = d_index[list_ind];

    float disp, mag, low, high;
    float EPS = 1e-5;
    int nump = 4;
    int size = w_size/nump;
    float inv = 1/(EPS * EPS);

    for(int i = 0; i < size; ++i){
        int j = int(d_w[nump * i]); //dimension to act on; x:0, y:1; z:2
        mag = d_w[nump * i + 1];
        low = d_w[nump * i + 2];
        high = d_w[nump * i + 3];
        int pi = ind * D + j;
        float my_x = x[pi];

        if (my_x <= low + EPS){
            x[pi] = low + EPS;
            f[pi] += mag * inv;
            v[pi] = abs(pi);
        }
        else {
            disp = my_x - low;
            f[pi] += mag/(disp * disp);
        }

        if (my_x >= high - EPS) {
            x[pi] = high - EPS;
            f[pi] -= mag * inv;
            v[pi] = -abs(pi);
        }
        else {
            disp = high - my_x;
            f[pi] -= mag/(disp * disp);
        }
    }
}

__global__ void d_wall_exp(
    float* f,
    float* x,
    float* v,
    thrust::device_ptr<float> d_w,
    const int w_size,
    thrust::device_ptr<int> d_index,
    const int ns,
    const int D) {

    int list_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (list_ind >= ns)
        return;

    int ind = d_index[list_ind];

    float disp, mag, low, high;
    float EPS = 1e-5;
    int nump = 4;
    int size = w_size/nump;

    for(int i = 0; i < size; ++i){
        int j = int(d_w[nump * i]); //dimension to act on; x:0, y:1; z:2
        mag = d_w[nump * i + 1];
        low = d_w[nump * i + 2];
        high = d_w[nump * i + 3];
        int pi = ind * D + j;
        float my_x = x[pi];

        if (my_x <= low + EPS){
            x[pi] = low + EPS;
            f[pi] += mag;
            v[pi] = abs(pi);
        }
        else {
            disp = my_x - low;
            f[pi] += mag * exp(-disp);
        }

        if (my_x >= high - EPS) {
            x[pi] = high - EPS;
            f[pi] -= mag;
            v[pi] = -abs(v[pi]);
        }
        else {
            disp = high - my_x;
            f[pi] -= mag * exp(-disp);
        }
    }
}
