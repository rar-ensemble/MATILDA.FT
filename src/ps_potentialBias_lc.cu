// Copyright (c) 2025 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "ps_potentialBias_lc.h"
#include "PS_Box.h"

// Forward declarations for kernels reused from device_utils.cu
__device__ float d_pbc_dr2f(float*, const float*, const float*, const float*, const float*, const int);
__global__ void d_assignFloatVal(float*, float, int);
__global__ void d_multiplyCpxByCpx(cuComplex*, const cuComplex*, const cuComplex*, int);
__global__ void d_multiplyCpxDirByCpx(cuComplex*, const cuComplex*, const cuComplex*, int, int, int);



NBBiasLC::NBBiasLC() {}

NBBiasLC::NBBiasLC(std::istringstream& iss, PS_Box* box) : PS_Potential(iss, box) {
    iss >> grpI;
    grpJ = grpI;

    iss >> Ao;

    iss >> filename;

    int dim = mybox->returnDimension();

    rmin = (float*) malloc(dim * sizeof(float));
    rmax = (float*) malloc(dim * sizeof(float));

    cudaMalloc(&d_rmin, dim*sizeof(float));
    cudaMalloc(&d_rmax, dim*sizeof(float));

    float umag = 0.0;

    // read in u0 and normalize 
    for ( int j=0 ; j<dim ; j++ ) {
        iss >> u0[j];
        umag += u0[j] * u0[j];
    }

    umag = sqrtf(umag);

    for ( int j=0 ; j<dim ; j++ ) {
        u0[j] = u0[j] / umag;

        rmin[j] = 0.0;
        rmax[j] = mybox->L[j];
    }

    // optional arguments
    while (iss.tellg() != -1) {
        std::string s1;
        iss >> s1;

        if ( s1 == "xrange" ) {
            iss >> rmin[0];
            iss >> rmax[0];
        }

        if ( s1 == "yrange" ) {
            iss >> rmin[1];
            iss >> rmax[1];
        }

        else if ( s1 == "zrange" ) {
            iss >> rmin[2];
            iss >> rmax[2];
        }
    }

    cudaMemcpy(d_rmin, rmin, dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_rmax, rmax, dim*sizeof(float), cudaMemcpyHostToDevice);
}

NBBiasLC::~NBBiasLC() {
    free(MS_pair);
    free(ms_u);
    free(ms_S);
    free(S0);
    free(ener);

    cudaFree(d_MS_pair);
    cudaFree(d_ms_u);
    cudaFree(d_ms_S);
    cudaFree(d_S0);
    cudaFree(d_ener);
}


// ─────────────────────────────────────────────────────────────────────────────
// initializePotential
// ─────────────────────────────────────────────────────────────────────────────
void NBBiasLC::initializePotential() {
    // Base class: resolves Iind/Jind, enables grid forces on both groups
    PS_Potential::initializePotential();

    int nstot = mybox->nstot;
    int Dim   = mybox->returnDimension();

    // ── Allocate host arrays ──────────────────────────────────────────────
    MS_pair          = (int*)   malloc(nstot * sizeof(int));
    ms_u             = (float*) malloc(Dim * nstot * sizeof(float));
    ms_S             = (float*) malloc(Dim * Dim * nstot * sizeof(float));
    S0               = (float*) malloc(Dim * Dim * sizeof(float));
    ener             = (float*) malloc(nstot * sizeof(float));

    // ── Allocate device arrays ────────────────────────────────────────────
    cudaMalloc(&d_MS_pair,  nstot * sizeof(int));
    cudaMalloc(&d_ms_u,     Dim * nstot * sizeof(float));
    cudaMalloc(&d_ms_S,     Dim * Dim * nstot * sizeof(float));
    cudaMalloc(&d_S0,       Dim * Dim * sizeof(float));
    cudaMalloc(&d_ener,     nstot * sizeof(float));
    check_cudaError("NBBiasLC: device allocation");

    // ── Initialize partner list to -1 (inactive) ─────────────────────────
    for (int i = 0; i < nstot; i++) MS_pair[i] = -1;

    // ── Read lc_file ─────────────────────────────────────────────────────
    read_lc_file(filename);

    // Initialize S0 tensor
    for ( int i=0 ; i<Dim ; i++ ) {
        for ( int j=0 ; j<Dim ; j++ ) {
            S0[i*Dim + j] = u0[i] * u0[j];
        }

        S0[i*Dim + i] -= 1.0 / float(Dim);
    }

    cudaMemcpy(d_S0, S0, Dim * Dim * sizeof(float), cudaMemcpyHostToDevice);
    check_cudaError("NBBiasLC: S0 upload");

    std::cout << "u0: " ; 
    for ( int j=0; j<Dim ; j++) std::cout << u0[j] << " ";
    std::cout << "\n";

    std::cout << "S0:\n";
    for ( int i=0 ; i<Dim ; i++ ) {
        for ( int j=0 ; j<Dim ; j++ ) {
            std::cout << S0[i*Dim+j] << " " ;
        }
        std::cout << "\n";
    }
    std::cout << std::endl;
}


// ─────────────────────────────────────────────────────────────────────────────
// read_lc_file  — reads partner-pair list for orientation vectors
// File format:
//   <nms>
//   <index> <id1> <id2>   (1-indexed; sets MS_pair[id1-1] = id2-1)
// ─────────────────────────────────────────────────────────────────────────────
void NBBiasLC::read_lc_file(std::string name) {
    FILE *inp = fopen(name.c_str(), "r");
    if (inp == NULL)
        die("NBBiasLC: lc_file not found: " + name);

    (void)!fscanf(inp, "%d\n", &nms);

    int id1, id2, di;
    for (int i = 0; i < nms; i++) {
        (void)!fscanf(inp, "%d %d %d\n", &di, &id1, &id2);
        MS_pair[id1 - 1] = id2 - 1;    // convert 1-indexed → 0-indexed
    }
    fclose(inp);

    cudaMemcpy(d_MS_pair, MS_pair, mybox->nstot * sizeof(int), cudaMemcpyHostToDevice);
    
    check_cudaError("NBBiasLC: MS_pair upload");


}


// ─────────────────────────────────────────────────────────────────────────────
// CalcSTensors
// Computes per-particle u vectors and S tensors, then maps S to the grid.
// ─────────────────────────────────────────────────────────────────────────────
void NBBiasLC::CalcSTensors() {
    int nstot = mybox->nstot;
    int Dim   = mybox->returnDimension();

    d_calcParticleSTensors<<<mybox->nsGrid, mybox->nsBlock>>>(
        d_ms_u, d_ms_S, mybox->d_x, d_MS_pair,
        mybox->d_L, mybox->d_Lh, Dim, nstot);
    check_cudaError("NBBiasLC: d_calcParticleSTensors");

}


// ─────────────────────────────────────────────────────────────────────────────
// CalcForces
// ─────────────────────────────────────────────────────────────────────────────
void NBBiasLC::CalcForces() {
    CalcSTensors();

    int nstot = mybox->nstot;
    int Dim   = mybox->returnDimension();

    d_accumulateMSBiasForce<<<mybox->nsGrid, mybox->nsBlock>>>(
        mybox->d_f, mybox->d_x, d_MS_pair, d_S0, d_ms_u, d_rmin, d_rmax,
        Ao, nstot, mybox->d_L, mybox->d_Lh, Dim);
    
    check_cudaError("NBBiasLC: accumulateMSForce2");

}// CalcForces()


// ─────────────────────────────────────────────────────────────────────────────
// CalcEnergy
// Assumes CalcForces() was called this step so d_tmp_tensor = (u⊗S)(r)
// and d_S_field = S(r).
// E = -∫ (u⊗S)(r) : S(r) dr
// ─────────────────────────────────────────────────────────────────────────────
float NBBiasLC::CalcEnergy() {

    int ns = mybox->nstot;
    int Dim = mybox->returnDimension();


    d_computeLCBiasEnergyPerPartic<<<mybox->nsGrid, mybox->nsBlock>>>(
        d_ener, mybox->d_x, d_rmin, d_rmax, d_ms_S, d_S0, ns, Dim);

    check_cudaError("NBBiasLC: d_doubleDotTensorFields");

    energy = -Ao * mybox->sumDeviceArray(d_ener, mybox->nsBlock, ns);

    check_cudaError("NBBiasLC::CalcEnergy end");

    return energy;
}


// ═════════════════════════════════════════════════════════════════════════════
// CUDA KERNELS
// ═════════════════════════════════════════════════════════════════════════════

// Element-wise double dot product of two tensor fields: out[i] = A[i]:B[i]
__global__ void d_computeLCBiasEnergyPerPartic(
    float* e,
    const float* x,     // [Dim*nstot]
    const float* rmin,  // [Dim]
    const float* rmax,  // [Dim]
    const float* Si,    // [Dim²*nstot]
    const float* S0,    // [Dim²]
    const int nstot,
    const int Dim
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= nstot) return;
    for ( int j=0 ; j<Dim ; j++ ) {
        if ( x[id*Dim+j] < rmin[j] ) return;
        else if ( x[id*Dim+j] > rmax[j] ) return;
    }

    float ddot = 0.0f;
    int base = id * Dim * Dim;
    for (int i = 0; i < Dim; i++)
        for (int j = 0; j < Dim; j++)
            ddot += Si[base + i * Dim + j] * S0[i * Dim + j];

    e[id] = ddot;
}


// Force contribution 2: derivative of S tensor w.r.t. particle position.
// Acts on both particles involved in the definition of u.
__global__ void d_accumulateMSBiasForce(
    float* f,               // [ns*Dim]
    const float* x,         // [ns*Dim]
    const int* upartner,    // [ns]
    const float* s0,        // [Dim²]
    const float* ms_u,      // [Dim*ns]
    const float* rmin,      // [Dim]
    const float* rmax,      // [Dim]
    const float Ao,         // Force magnitude
    const int ns,
    const float* L,
    const float* Lh,
    const int Dim
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ns) return;
    if (upartner[id] < 0 || upartner[id] >= ns) return;

    int id1 = upartner[id];

    float ri[3], rj[3], dr[3], fi[3];
    for (int j = 0; j < Dim; j++) {
        ri[j] = x[id  * Dim + j];
        if ( ri[j] < rmin[j] ) return;
        else if ( ri[j] > rmax[j] ) return;

        rj[j] = x[id1 * Dim + j];
        fi[j] = 0.0f;
    }

    // Identity tensor
    float I[9] = {0};
    for (int j = 0; j < Dim; j++) I[j * Dim + j] = 1.0f;

    // dr = rj - ri (PBC)
    float mdr2 = d_pbc_dr2f(dr, rj, ri, L, Lh, Dim);
    float mdr  = sqrtf(mdr2);

    // For each spatial component 'a' of the force on particle id:
    for (int a = 0; a < Dim; a++) {

        // dudra[j] = d(u[j])/d(ri[a])  with u = dr/|dr|
        float dudra[3];
        for (int j = 0; j < Dim; j++) {
            dudra[j] = dr[j] * dr[a] / mdr2;
            if (j == a) dudra[j] -= 1.0f;
            dudra[j] /= mdr;
        }

        // T = (I u + u I) · dudra   (rank-2 tensor)
        float T[9] = {0};
        for (int j = 0; j < Dim; j++) {
            for (int k = 0; k < Dim; k++) {
                for (int mm = 0; mm < Dim; mm++) {
                    T[j * Dim + k] += (I[j * Dim + mm] * ms_u[id * Dim + k]
                                     + ms_u[id * Dim + j] * I[k * Dim + mm])
                                     * dudra[mm];
                }
            }
        }


        float dotsum = 0.0f;
        for (int j = 0; j < Dim; j++)
            for (int k = 0; k < Dim; k++)
                dotsum += T[j * Dim + k] * s0[j * Dim + k];

        fi[a] += dotsum * Ao;
        
    }// for a=0:Dim

    // Newton's 3rd law: partner gets opposite force
    for (int a = 0; a < Dim; a++) {
        atomicAdd(&f[id  * Dim + a],  fi[a]);
        atomicAdd(&f[id1 * Dim + a], -fi[a]);
    }
}

