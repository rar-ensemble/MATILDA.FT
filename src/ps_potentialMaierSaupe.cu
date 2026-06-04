// Copyright (c) 2025 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "ps_potentialMaierSaupe.h"
#include "PS_Box.h"

// using namespace Eigen;

// Forward declarations for kernels reused from device_utils.cu
__device__ float d_pbc_dr2f(float*, const float*, const float*, const float*, const float*, const int);
__global__ void d_assignFloatVal(float*, float, int);
__global__ void d_multiplyCpxByCpx(cuComplex*, const cuComplex*, const cuComplex*, int);
__global__ void d_multiplyCpxDirByCpx(cuComplex*, const cuComplex*, const cuComplex*, int, int, int);

int NBMaier::num = 0;


NBMaier::NBMaier() {}

NBMaier::NBMaier(std::istringstream& iss, PS_Box* box) : PS_Potential(iss, box) {
    iss >> grpI;
    iss >> grpJ;
    iss >> Ao;
    iss >> sig2;
    sig2 *= sig2;
    iss >> filename;
}

NBMaier::~NBMaier() {
    free(MS_pair);
    free(ms_u);
    free(ms_S);
    free(S_field);
    free(h_Dim_Dim_tensor);

    cudaFree(d_MS_pair);
    cudaFree(d_ms_u);
    cudaFree(d_ms_S);
    cudaFree(d_S_field);
    cudaFree(d_tmp_tensor);
    cudaFree(d_Dim_Dim_tensor);
}


// ─────────────────────────────────────────────────────────────────────────────
// initializePotential
// ─────────────────────────────────────────────────────────────────────────────
void NBMaier::initializePotential() {
    // Base class: resolves Iind/Jind, enables grid forces on both groups
    PS_Potential::initializePotential();

    int nstot = mybox->nstot;
    int M     = mybox->M;
    int Dim   = mybox->returnDimension();

    // ── Allocate host arrays ──────────────────────────────────────────────
    MS_pair          = (int*)   malloc(nstot * sizeof(int));
    ms_u             = (float*) malloc(Dim * nstot * sizeof(float));
    ms_S             = (float*) malloc(Dim * Dim * nstot * sizeof(float));
    S_field          = (float*) malloc(Dim * Dim * M * sizeof(float));
    h_Dim_Dim_tensor = (float*) malloc(Dim * Dim * sizeof(float));

    // ── Allocate device arrays ────────────────────────────────────────────
    cudaMalloc(&d_MS_pair,          nstot * sizeof(int));
    cudaMalloc(&d_ms_u,             Dim * nstot * sizeof(float));
    cudaMalloc(&d_ms_S,             Dim * Dim * nstot * sizeof(float));
    cudaMalloc(&d_S_field,          Dim * Dim * M * sizeof(float));
    cudaMalloc(&d_tmp_tensor,       Dim * Dim * M * sizeof(float));
    cudaMalloc(&d_Dim_Dim_tensor,   Dim * Dim * sizeof(float));
    check_cudaError("NBMaier: device allocation");

    // Zero tensor fields so the first CalcEnergy() (before any CalcForces()) is safe.
    cudaMemset(d_S_field,    0, Dim * Dim * M * sizeof(float));
    cudaMemset(d_tmp_tensor, 0, Dim * Dim * M * sizeof(float));
    cudaMemset(d_ms_S,       0, Dim * Dim * nstot * sizeof(float));
    check_cudaError("NBMaier: zero tensor arrays");

    // ── Initialize partner list to -1 (inactive) ─────────────────────────
    for (int i = 0; i < nstot; i++) MS_pair[i] = -1;

    // ── Read lc_file ─────────────────────────────────────────────────────
    read_lc_file(filename);

    // ── Build uk and fk in k-space (Gaussian kernel) ─────────────────────
    // uk[i]        = Ao * exp(-k²σ²/2)
    // fk[i*Dim+j]  = -i * kj * uk[i]   (gradient in k-space)
    std::complex<float> I(0.0f, 1.0f);
    float kv[3];
    for (int i = 0; i < M; i++) {
        float k2 = mybox->get_kD(i, kv);
        uk[i] = Ao * expf(-k2 * sig2 / 2.0f);
        for (int j = 0; j < Dim; j++) {
            fk[i * Dim + j] = -I * kv[j] * uk[i];
        }
    }

    cudaMemcpy(d_uk, uk, M * sizeof(std::complex<float>), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fk, fk, M * Dim * sizeof(std::complex<float>), cudaMemcpyHostToDevice);
    check_cudaError("NBMaier: k-space potential upload");

    std::cout << "  Maier-Saupe potential initialized: Ao=" << Ao
              << " sigma=" << sqrtf(sig2)
              << " nms=" << nms << std::endl;
}


// ─────────────────────────────────────────────────────────────────────────────
// read_lc_file  — reads partner-pair list for orientation vectors
// File format:
//   <nms>
//   <index> <id1> <id2>   (1-indexed; sets MS_pair[id1-1] = id2-1)
// ─────────────────────────────────────────────────────────────────────────────
void NBMaier::read_lc_file(std::string name) {
    FILE *inp = fopen(name.c_str(), "r");
    if (inp == NULL)
        die("NBMaier: lc_file not found: " + name);

    (void)!fscanf(inp, "%d\n", &nms);

    int id1, id2, di;
    for (int i = 0; i < nms; i++) {
        (void)!fscanf(inp, "%d %d %d\n", &di, &id1, &id2);
        MS_pair[id1 - 1] = id2 - 1;    // convert 1-indexed → 0-indexed
    }
    fclose(inp);

    cudaMemcpy(d_MS_pair, MS_pair, mybox->nstot * sizeof(int), cudaMemcpyHostToDevice);
    
    check_cudaError("NBMaier: MS_pair upload");


}


// ─────────────────────────────────────────────────────────────────────────────
// CalcSTensors
// Computes per-particle u vectors and S tensors, then maps S to the grid.
// ─────────────────────────────────────────────────────────────────────────────
void NBMaier::CalcSTensors() {
    int nstot = mybox->nstot;
    int M     = mybox->M;
    int Dim   = mybox->returnDimension();

    d_calcParticleSTensors<<<mybox->nsGrid, mybox->nsBlock>>>(
        d_ms_u, d_ms_S, mybox->d_x, d_MS_pair,
        mybox->d_L, mybox->d_Lh, Dim, nstot);
    check_cudaError("NBMaier: d_calcParticleSTensors");

    int DD = Dim * Dim;
    int DDM_Grid = (int)ceil((float)(DD * M) / mybox->M_Block);
    d_assignFloatVal<<<DDM_Grid, mybox->M_Block>>>(d_S_field, 0.0f, DD * M);
    check_cudaError("NBMaier: d_assignFloatVal S_field");

    d_mapFieldSTensors<<<mybox->nsGrid, mybox->nsBlock>>>(
        d_S_field, d_MS_pair, d_ms_S,
        mybox->d_gridW, mybox->d_gridInds,
        nstot, mybox->gridPerPartic, Dim);
    check_cudaError("NBMaier: d_mapFieldSTensors");
}


// ─────────────────────────────────────────────────────────────────────────────
// CalcForces
// Two contributions:
//   1. Force from ∇u(r):  F1 ∝ S(r) : ∇(u⊗S)(r)
//   2. Force from dS/dr:  F2 ∝ (u⊗S)(r) : dS/dr
// ─────────────────────────────────────────────────────────────────────────────
void NBMaier::CalcForces() {
    CalcSTensors();

    int nstot = mybox->nstot;
    int M     = mybox->M;
    int Dim   = mybox->returnDimension();

    cuComplex *d_cpxA = mybox->d_cpxAlex;
    cuComplex *d_cpxG = mybox->d_cpxGabe;

    // ── Contribution 1: grad u term ───────────────────────────────────────
    // For each output direction j, convolve each S_km with fk[j], then
    // accumulate the tensor-contracted force onto particles.
    for (int j = 0; j < Dim; j++) {

        for (int k = 0; k < Dim; k++) {
            for (int m = k; m < Dim; m++) {

                // Extract S_km into complex scratch (imag = 0)
                d_extractTensorComponent<<<mybox->M_Grid, mybox->M_Block>>>(
                    d_cpxA, d_S_field, k, m, M, Dim);

                // d_cpxG = FFT(S_km)
                mybox->cufftWrapperSingle(d_cpxA, d_cpxG, 1);
                check_cudaError("NBMaier: FFT(S_km) – force1");

                // d_cpxA = fk[j] * FFT(S_km)
                d_multiplyCpxDirByCpx<<<mybox->M_Grid, mybox->M_Block>>>(
                    d_cpxA, d_fk, d_cpxG, j, Dim, M);

                // d_cpxA = IFFT(fk[j] * FFT(S_km))
                mybox->cufftWrapperSingle(d_cpxA, d_cpxA, -1);
                check_cudaError("NBMaier: IFFT – force1");

                // Store real part into tmp_tensor[k,m]
                d_storeTensorComponent<<<mybox->M_Grid, mybox->M_Block>>>(
                    d_tmp_tensor, d_cpxA, k, m, M, Dim);

                // Exploit symmetry: S_mk = S_km
                if (k != m) {
                    d_storeTensorComponent<<<mybox->M_Grid, mybox->M_Block>>>(
                        d_tmp_tensor, d_cpxA, m, k, M, Dim);
                }

            }// m
        }// k

        // Accumulate force contribution 1 in direction j
        d_accumulateMSForce1<<<mybox->nsGrid, mybox->nsBlock>>>(
            mybox->d_f, d_MS_pair, d_tmp_tensor, d_ms_S,
            mybox->d_gridW, mybox->d_gridInds,
            (float)mybox->gvol, j, mybox->gridPerPartic, nstot, Dim);
        check_cudaError("NBMaier: accumulateMSForce1");

    }// j


    // ── Contribution 2: dS/dr term ────────────────────────────────────────
    // Convolve each S_km with u(k), assemble the full (u⊗S) tensor field,
    // then contract against dS/dr at each MS particle.
    for (int k = 0; k < Dim; k++) {
        for (int m = k; m < Dim; m++) {

            d_extractTensorComponent<<<mybox->M_Grid, mybox->M_Block>>>(
                d_cpxA, d_S_field, k, m, M, Dim);

            // d_cpxG = FFT(S_km)
            mybox->cufftWrapperSingle(d_cpxA, d_cpxG, 1);
            check_cudaError("NBMaier: FFT(S_km) – force2");

            // d_cpxA = d_uk * FFT(S_km)
            d_multiplyCpxByCpx<<<mybox->M_Grid, mybox->M_Block>>>(
                d_cpxA, d_uk, d_cpxG, M);

            // d_cpxA = IFFT(uk * FFT(S_km)) = (u ⊗ S_km)(r)
            mybox->cufftWrapperSingle(d_cpxA, d_cpxA, -1);
            check_cudaError("NBMaier: IFFT – force2");

            d_storeTensorComponent<<<mybox->M_Grid, mybox->M_Block>>>(
                d_tmp_tensor, d_cpxA, k, m, M, Dim);

            if (k != m) {
                d_storeTensorComponent<<<mybox->M_Grid, mybox->M_Block>>>(
                    d_tmp_tensor, d_cpxA, m, k, M, Dim);
            }

        }// m
    }// k

    d_accumulateMSForce2<<<mybox->nsGrid, mybox->nsBlock>>>(
        mybox->d_f, mybox->d_x, d_MS_pair, d_tmp_tensor, d_ms_u,
        mybox->d_gridW, mybox->d_gridInds,
        (float)mybox->gvol, mybox->gridPerPartic, nstot,
        mybox->d_L, mybox->d_Lh, Dim);
    check_cudaError("NBMaier: accumulateMSForce2");


}// CalcForces()


// ─────────────────────────────────────────────────────────────────────────────
// CalcEnergy
// Assumes CalcForces() was called this step so d_tmp_tensor = (u⊗S)(r)
// and d_S_field = S(r).
// E = -∫ (u⊗S)(r) : S(r) dr
// ─────────────────────────────────────────────────────────────────────────────
float NBMaier::CalcEnergy() {

    int M = mybox->M;
    int Dim = mybox->returnDimension();


    d_doubleDotTensorFields<<<mybox->M_Grid, mybox->M_Block>>>(
        mybox->d_Gabe, d_tmp_tensor, d_S_field, M, Dim);
    check_cudaError("NBMaier: d_doubleDotTensorFields");

    energy = -0.5f * (float)mybox->gvol *
              mybox->sumDeviceArray(mybox->d_Gabe, mybox->M_Block, M);

    check_cudaError("NBMaier::CalcEnergy end");

    return energy;
}

void NBMaier::initBinaryOutput() {
    std::string name;
    name = "Sfield-" + grpI + "-" + grpJ + std::string(".bin");
    mybox->initBinaryDataFile(name);
}


void NBMaier::writeBinaryOutput() {
    std::string name;
    name = "Sfield-" + grpI + "-" + grpJ + std::string(".bin");

    int d = mybox->returnDimension();

    cudaMemcpy(S_field, d_S_field, d*d*mybox->M*sizeof(float), cudaMemcpyDeviceToHost);

    mybox->writeBinaryTensorData(name, S_field);
}









// ═════════════════════════════════════════════════════════════════════════════
// CUDA KERNELS
// ═════════════════════════════════════════════════════════════════════════════

// Compute per-particle orientation vectors u and S tensors from the partner list.
// u = (r_partner - r_self) / |r_partner - r_self|  (PBC corrected)
// S[j,k] = u[j]*u[k] - delta[j,k]/Dim
__global__ void d_calcParticleSTensors(
    float* ms_u,            // [Dim*ns] orientation vectors
    float* ms_S,            // [Dim²*ns] S tensors
    const float* x,         // [Dim*ns] positions
    const int* upartner,    // [ns] partner index (-1 = inactive)
    const float* L,         // [Dim] box lengths
    const float* Lh,        // [Dim] half box lengths
    const int Dim,
    const int ns
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ns) return;
    if (upartner[id] < 0 || upartner[id] >= ns) return;

    int id2 = upartner[id];

    float r1[3], r2[3], dr[3];

    for (int j = 0; j < Dim; j++) {
        r1[j] = x[id  * Dim + j];
        r2[j] = x[id2 * Dim + j];
    }

    // dr = r2 - r1 with PBC;  d_pbc_dr2f computes dr = ri - rj
    float mdr2 = d_pbc_dr2f(dr, r2, r1, L, Lh, Dim);
    float mdr  = sqrtf(mdr2);

    for (int j = 0; j < Dim; j++)
        ms_u[id * Dim + j] = dr[j] / mdr;

    int base = id * Dim * Dim;
    for (int j = 0; j < Dim; j++) {
        for (int k = 0; k < Dim; k++)
            ms_S[base + j * Dim + k] = ms_u[id * Dim + j] * ms_u[id * Dim + k];
        ms_S[base + j * Dim + j] -= 1.0f / float(Dim);
    }
}


// Scatter per-particle S tensors onto the density grid using precomputed weights.
__global__ void d_mapFieldSTensors(
    float* field_S,         // [Dim²*M] S tensor field (accumulated)
    const int* upartner,    // [ns] partner index
    const float* ms_S,      // [Dim²*ns] particle S tensors
    const float* grid_W,    // [ns*gridPerPartic] weights
    const int* grid_inds,   // [ns*gridPerPartic] grid indices
    const int ns,
    const int grid_per_partic,
    const int Dim
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ns) return;
    if (upartner[id] < 0) return;

    for (int i = 0; i < grid_per_partic; i++) {
        float W3    = grid_W   [id * grid_per_partic + i];
        int space   = grid_inds[id * grid_per_partic + i];

        for (int j = 0; j < Dim; j++) {
            for (int k = 0; k < Dim; k++) {
                atomicAdd(&field_S[space * Dim * Dim + j * Dim + k],
                          W3 * ms_S[id * Dim * Dim + j * Dim + k]);
            }
        }
    }
}


// Load one (ii,jj) component of the tensor field into a complex array (imag=0).
__global__ void d_extractTensorComponent(
    cuComplex* dest,        // [M] output
    const float* Tfield,    // [Dim²*M] tensor field
    const int ii, const int jj,
    const int M, const int Dim
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= M) return;
    dest[id].x = Tfield[id * Dim * Dim + ii * Dim + jj];
    dest[id].y = 0.0f;
}


// Store the real part of a complex array into the (ii,jj) component of a tensor field.
__global__ void d_storeTensorComponent(
    float* Tfield,          // [Dim²*M] tensor field
    const cuComplex* src,   // [M] source (real part used)
    const int ii, const int jj,
    const int M, const int Dim
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= M) return;
    Tfield[id * Dim * Dim + ii * Dim + jj] = src[id].x;
}


// Force contribution 1: interpolate the convolved tensor field to particle positions
// and contract against the particle's own S tensor.
// F1[id, dir] += gvol * sum_m W3_m * (-tensorField[m] : ms_S[id])
__global__ void d_accumulateMSForce1(
    float* f,               // [ns*Dim] particle forces (accumulated)
    const int* upartner,    // [ns]
    const float* tensorField, // [Dim²*M] (u⊗S)(r) from contribution-1 loop
    const float* ms_S,      // [Dim²*ns]
    const float* grid_W,    // [ns*gridPerPartic]
    const int* grid_inds,   // [ns*gridPerPartic]
    const float gvol,
    const int dir,          // force component (j in outer loop)
    const int grid_per_partic,
    const int ns,
    const int Dim
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ns) return;
    if (upartner[id] < 0) return;

    int pbase = id * Dim * Dim;
    float fval = 0.0f;

    for (int m = 0; m < grid_per_partic; m++) {
        int gind  = id * grid_per_partic + m;
        int Mind  = grid_inds[gind];
        float W3  = grid_W[gind];

        float dotsum = 0.0f;
        for (int j = 0; j < Dim; j++) {
            for (int k = 0; k < Dim; k++) {
                // Negative sign: potential is -Ao*Gaussian so force sign flips
                dotsum += -tensorField[Mind * Dim * Dim + j * Dim + k]
                          * ms_S[pbase + j * Dim + k];
            }
        }
        fval += dotsum * W3 * gvol;
    }

    f[id * Dim + dir] += fval;
}


// Force contribution 2: derivative of S tensor w.r.t. particle position.
// Acts on both particles involved in the definition of u.
__global__ void d_accumulateMSForce2(
    float* f,               // [ns*Dim]
    const float* x,         // [ns*Dim]
    const int* upartner,    // [ns]
    const float* tensorField, // [Dim²*M] (u⊗S)(r)
    const float* ms_u,      // [Dim*ns]
    const float* grid_W,    // [ns*gridPerPartic]
    const int* grid_inds,   // [ns*gridPerPartic]
    const float gvol,
    const int grid_per_partic,
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

        // Interpolate (u⊗S)(r) to particle position and contract with T
        for (int mm = 0; mm < grid_per_partic; mm++) {
            int gind  = id * grid_per_partic + mm;
            int Mind  = grid_inds[gind];
            float W3  = grid_W[gind];

            float dotsum = 0.0f;
            for (int j = 0; j < Dim; j++)
                for (int k = 0; k < Dim; k++)
                    dotsum += T[j * Dim + k]
                              * tensorField[Mind * Dim * Dim + j * Dim + k];

            fi[a] += dotsum * W3 * gvol;
        }
    }

    // Newton's 3rd law: partner gets opposite force
    for (int a = 0; a < Dim; a++) {
        atomicAdd(&f[id  * Dim + a],  fi[a]);
        atomicAdd(&f[id1 * Dim + a], -fi[a]);
    }
}


// Element-wise double dot product of two tensor fields: out[i] = A[i]:B[i]
__global__ void d_doubleDotTensorFields(
    float* out,
    const float* S1,    // [Dim²*M]
    const float* S2,    // [Dim²*M]
    const int M,
    const int Dim
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= M) return;

    float ddot = 0.0f;
    int base = id * Dim * Dim;
    for (int i = 0; i < Dim; i++)
        for (int j = 0; j < Dim; j++)
            ddot += S1[base + i * Dim + j] * S2[base + i * Dim + j];
    out[id] = ddot;
}


// Accumulate all per-particle S tensors into a [Dim²] sum array.
// Only active (upartner >= 0) particles contribute.
__global__ void d_SumAndAverageSTensors(
    const float* ms_S,      // [Dim²*ns]
    float* sumTensor,       // [Dim²]
    const int* upartner,    // [ns]
    const int Dim,
    const int ns
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ns) return;
    if (upartner[id] < 0) return;

    int DD = Dim * Dim;
    for (int j = 0; j < DD; j++)
        atomicAdd(&sumTensor[j], ms_S[id * DD + j]);
}
