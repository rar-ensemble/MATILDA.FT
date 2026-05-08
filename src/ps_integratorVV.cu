// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "ps_integratorVV.h"
#include "PS_Box.h"


VV::VV(std::istringstream& iss, PS_Box* box) : Integrator(iss, box), d_invMass(nullptr) {}

VV::~VV() {
    cudaFree(d_invMass);
}

// Precompute per-type inverse masses so kernels avoid division every step.
// Velocities are already zeroed: v[] was calloc'd on the host and copied by
// sendAllHostToDevice before finishInitialization is called.
void VV::finishInitialization() {
    const int nTypes = mybox->nTypes;
    float* h_invMass = (float*) malloc(nTypes * sizeof(float));
    for (int i = 0; i < nTypes; i++)
        h_invMass[i] = 1.0f / mybox->speciesMass[i];

    cudaMalloc(&d_invMass, nTypes * sizeof(float));
    cudaMemcpy(d_invMass, h_invMass, nTypes * sizeof(float), cudaMemcpyHostToDevice);
    free(h_invMass);
}


// First half-step:
//   v(t + dt/2) = v(t) + (dt / 2m) * f(t)
//   x(t + dt)   = x(t) + dt * v(t + dt/2)
//   Apply PBC to x.
__global__ void d_VV_integrate_1(
    float* x,               // [nstot*Dim]
    float* v,               // [nstot*Dim]
    const float* f,         // [nstot*Dim]
    const float* d_invMass, // [nTypes] 1/mass per species
    const int* typ,         // [nstot] particle type index
    const float* L,         // [Dim] box dimensions
    const int* d_index,     // [ns] site list
    const int ns,
    const int Dim,
    const float half_delt)  // delt / 2
{
    const int list_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (list_ind >= ns) return;

    const int ind = d_index[list_ind];
    const float inv_m = d_invMass[typ[ind]];

    for (int j = 0; j < Dim; j++) {
        const int aind = ind * Dim + j;

        // half-kick
        v[aind] += half_delt * inv_m * f[aind];

        // full position update
        float xnew = x[aind] + (half_delt * 2.0f) * v[aind];

        // PBC wrap
        if (xnew >= L[j]) xnew -= L[j];
        else if (xnew < 0.0f) xnew += L[j];

        x[aind] = xnew;
    }
}


// Second half-step:
//   v(t + dt) = v(t + dt/2) + (dt / 2m) * f(t + dt)
__global__ void d_VV_integrate_2(
    float* v,               // [nstot*Dim]
    const float* f,         // [nstot*Dim]
    const float* d_invMass, // [nTypes] 1/mass per species
    const int* typ,         // [nstot] particle type index
    const int* d_index,     // [ns] site list
    const int ns,
    const int Dim,
    const float half_delt)
{
    const int list_ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (list_ind >= ns) return;

    const int ind = d_index[list_ind];
    const float inv_m = d_invMass[typ[ind]];

    for (int j = 0; j < Dim; j++) {
        const int aind = ind * Dim + j;
        v[aind] += half_delt * inv_m * f[aind];
    }
}


void VV::Integrate_1() {
    const int grid = mybox->psGroup[group_index].Grid;
    const int block = mybox->psGroup[group_index].Block;
    const int ns   = mybox->psGroup[group_index].nsites;
    const int Dim  = mybox->returnDimension();

    d_VV_integrate_1<<<grid, block>>>(
        mybox->d_x, mybox->d_v, mybox->d_f,
        d_invMass, mybox->d_intSpecies, mybox->d_L,
        mybox->psGroup[group_index].d_siteList,
        ns, Dim, delt * 0.5f);
}

void VV::Integrate_2() {
    const int grid = mybox->psGroup[group_index].Grid;
    const int block = mybox->psGroup[group_index].Block;
    const int ns   = mybox->psGroup[group_index].nsites;
    const int Dim  = mybox->returnDimension();

    d_VV_integrate_2<<<grid, block>>>(
        mybox->d_v, mybox->d_f,
        d_invMass, mybox->d_intSpecies,
        mybox->psGroup[group_index].d_siteList,
        ns, Dim, delt * 0.5f);
}
