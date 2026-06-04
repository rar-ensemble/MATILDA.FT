// Copyright (c) 2025 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "ps_potentialWall.h"
#include "PS_Box.h"

__global__ void d_wallHarmonicForce(float*, const float*, const int*,
    const int, const int, const float, const int, const float, const int);

__global__ void d_wallHarmonicEnergy(float*, const float*, const int*,
    const int, const int, const float, const int, const float, const int);


Wall::Wall(std::istringstream& iss, PS_Box* box) : PS_Potential(iss, box) {

    iss >> grpI;
    grpJ = grpI;

    std::string axis;
    iss >> axis;
    if      (axis == "x" || axis == "X") normalDim = 0;
    else if (axis == "y" || axis == "Y") normalDim = 1;
    else if (axis == "z" || axis == "Z") normalDim = 2;
    else {
        std::string err = "ps_potentialWall.cu: axis must be x, y, or z (got '" + axis + "')";
        die(err.c_str());
    }

    int Dim = mybox->returnDimension();
    if (normalDim >= Dim) {
        std::string err = "ps_potentialWall.cu: requested wall normal '" + axis +
                          "' but box is only " + std::to_string(Dim) + "-dimensional";
        die(err.c_str());
    }

    iss >> wallPos;
    iss >> dirSign;
    if (dirSign != 1 && dirSign != -1) {
        std::string err = "ps_potentialWall.cu: dirSign must be +1 or -1 (got " +
                          std::to_string(dirSign) + ")";
        die(err.c_str());
    }

    iss >> k;
}


void Wall::initializePotential() {

    PS_Potential::initializePotential();

    int ns = mybox->psGroup[Iind].nsites;
    cudaMalloc(&d_ener, ns * sizeof(float));
    check_cudaError("Wall: d_ener alloc");
}


void Wall::CalcForces() {

    int GRID  = mybox->psGroup[Iind].Grid;
    int BLOCK = mybox->psGroup[Iind].Block;
    int Dim   = mybox->returnDimension();
    int ns    = mybox->psGroup[Iind].nsites;

    d_wallHarmonicForce<<<GRID, BLOCK>>>(mybox->d_f, mybox->d_x,
        mybox->psGroup[Iind].d_siteList,
        ns, normalDim, wallPos, dirSign, k, Dim);
    check_cudaError("Wall: d_wallHarmonicForce");
}


float Wall::CalcEnergy() {

    int GRID  = mybox->psGroup[Iind].Grid;
    int BLOCK = mybox->psGroup[Iind].Block;
    int Dim   = mybox->returnDimension();
    int ns    = mybox->psGroup[Iind].nsites;

    d_wallHarmonicEnergy<<<GRID, BLOCK>>>(d_ener, mybox->d_x,
        mybox->psGroup[Iind].d_siteList,
        ns, normalDim, wallPos, dirSign, k, Dim);
    check_cudaError("Wall: d_wallHarmonicEnergy");

    this->energy = mybox->sumDeviceArray(d_ener, BLOCK, ns);
    return this->energy;
}


Wall::Wall() {}

Wall::~Wall() {
    cudaFree(d_ener);
}


// ═════════════════════════════════════════════════════════════════════════════
// CUDA KERNELS
// ═════════════════════════════════════════════════════════════════════════════

// F_n = k * (wallPos - x_n) when particle is on the forbidden side; 0 otherwise.
// "Forbidden side" means dirSign * (wallPos - x_n) > 0.
__global__ void d_wallHarmonicForce(
    float* f,
    const float* x,
    const int* sites,
    const int ns,
    const int normalDim,
    const float wallPos,
    const int dirSign,
    const float k,
    const int Dim
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ns) return;

    int pind = sites[id];
    float disp = wallPos - x[pind * Dim + normalDim];

    if ((float)dirSign * disp > 0.0f) {
        atomicAdd(&f[pind * Dim + normalDim], k * disp);
    }
}


__global__ void d_wallHarmonicEnergy(
    float* e,
    const float* x,
    const int* sites,
    const int ns,
    const int normalDim,
    const float wallPos,
    const int dirSign,
    const float k,
    const int Dim
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ns) return;

    int pind = sites[id];
    float disp = wallPos - x[pind * Dim + normalDim];

    e[id] = ((float)dirSign * disp > 0.0f) ? 0.5f * k * disp * disp : 0.0f;
}
