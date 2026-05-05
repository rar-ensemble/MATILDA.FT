// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "ps_potentialDPD.h"
#include "PS_Box.h"

__device__ float d_pbc_dr2f(float*, const float*, const float*, const float*, const float*, const int);


DPD::DPD(std::istringstream& iss, PS_Box* box) : PS_Potential(iss, box), nList(nullptr) {

    iss >> grpI;
    grpJ = grpI;

    iss >> gamma >> kT;
    sigma = sqrtf(2.0f * gamma * kT);

    if (mybox->integrators.size() > 0)
        delt = mybox->integrators[0]->delt;
    else
        delt = -1.0f;

    while (iss.tellg() != -1) {
        std::string word;
        iss >> word;
        if (word == "delt")
            iss >> delt;
        else
            die("Invalid keyword in dpd potential: " + word);
    }

    if (delt <= 0.0f)
        die("ps_potentialDPD: delt not set or invalid");
}


void DPD::initializePotential() {
    PS_Potential::initializePotential();

    for (auto* nl : mybox->neighborLists) {
        if (nl->groupInd == Iind) { nList = nl; break; }
    }
    if (!nList)
        die("dpd: no neighbor_list found for group " + grpI +
            " — add 'neighbor_list " + grpI + " <rcut>' before this potential");
}

float DPD::CalcEnergy() {
    return 0.0f;
}

void DPD::CalcForces() {
    const int Grid  = mybox->psGroup[Iind].Grid;
    const int Block = mybox->psGroup[Iind].Block;
    const int ns    = mybox->psGroup[Iind].nsites;
    const int Dim   = mybox->returnDimension();

    d_DPD_forces<<<Grid, Block>>>(
        mybox->d_f, mybox->d_v, mybox->d_x,
        mybox->d_L, mybox->d_Lh,
        nList->d_neighborList, nList->d_nNeighbors,
        mybox->psGroup[Iind].d_siteList,
        mybox->d_states,
        gamma, sigma, nList->rcut,
        1.0f / sqrtf(delt),
        nList->maxNeighbors, ns, Dim);
}


// Pairwise DPD dissipative + random forces.
// One thread per group particle (t = group-local index).
// Only processes pairs with j_global > i_global (half-pair) so each pair
// is handled exactly once and Newton's third law is enforced via atomicAdd.
// All writes to d_f use atomicAdd because particle i of one thread may be
// the j-target of another thread running concurrently.
__global__ void d_DPD_forces(
    float*       d_f,
    const float* d_v,
    const float* d_x,
    const float* d_L,
    const float* d_Lh,
    const int*   d_neighborList,
    const int*   d_nNeighbors,
    const int*   d_siteList,
    curandState* d_states,
    const float  gamma,
    const float  sigma,
    const float  rcut,
    const float  inv_sqrt_delt,
    const int    maxNeighbors,
    const int    nsites,
    const int    Dim)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nsites) return;

    const int i_global = d_siteList[t];
    curandState l_state = d_states[i_global];

    const int nn = d_nNeighbors[t];
    for (int k = 0; k < nn; k++) {
        const int j_global = d_neighborList[t * maxNeighbors + k];
        if (j_global <= i_global) continue;   // half-pair: process i < j only

        float dr[3];
        float r2 = d_pbc_dr2f(dr,
            d_x + i_global * Dim,
            d_x + j_global * Dim,
            d_L, d_Lh, Dim);

        if (r2 >= rcut * rcut) continue;

        const float r    = sqrtf(r2);
        const float w    = 1.0f - r / rcut;
        const float winv = 1.0f / r;          // used to normalize dr to r_hat

        // relative velocity projected onto r_hat
        float vdot = 0.0f;
        for (int d = 0; d < Dim; d++)
            vdot += (d_v[i_global * Dim + d] - d_v[j_global * Dim + d]) * (dr[d] * winv);

        const float xi   = curand_normal(&l_state);
        const float Fmag = -gamma * w * w * vdot + sigma * w * xi * inv_sqrt_delt;

        for (int d = 0; d < Dim; d++) {
            const float Fd = Fmag * (dr[d] * winv);   // dr[d]/r = r_hat component
            atomicAdd(&d_f[i_global * Dim + d],  Fd);
            atomicAdd(&d_f[j_global * Dim + d], -Fd);
        }
    }

    d_states[i_global] = l_state;
}
