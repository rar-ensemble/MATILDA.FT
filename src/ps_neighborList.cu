// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "PS_Box.h"
#include "ps_neighborList.h"
#include <cub/cub.cuh>
#include <cmath>
#include <iostream>

// Forward declaration from device_utils.cu
__device__ float d_pbc_dr2f(float*, const float*, const float*, const float*, const float*, const int);


// Assign each group particle to a cell and record its global ID.
// Thread t handles group particle siteList[t].
__global__ void d_assignCellIDs(
    int* cellID, int* particleID,
    const int* siteList,
    const float* d_x,
    float cellWidthX, float cellWidthY, float cellWidthZ,
    int nCellsX, int nCellsY, int nCellsZ,
    const float* d_L,
    int Dim, int nsites)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nsites) return;

    const int id = siteList[t];
    particleID[t] = id;

    // Compute integer cell index for each dimension
    int cx = (int)(d_x[id*Dim + 0] / cellWidthX);
    cx = cx < 0 ? 0 : (cx >= nCellsX ? nCellsX-1 : cx);

    int cy = 0, cz = 0;
    if (Dim >= 2) {
        cy = (int)(d_x[id*Dim + 1] / cellWidthY);
        cy = cy < 0 ? 0 : (cy >= nCellsY ? nCellsY-1 : cy);
    }
    if (Dim >= 3) {
        cz = (int)(d_x[id*Dim + 2] / cellWidthZ);
        cz = cz < 0 ? 0 : (cz >= nCellsZ ? nCellsZ-1 : cz);
    }

    cellID[t] = cx + nCellsX * (cy + nCellsY * cz);
}


// Mark the start and end of each cell in the sorted particle array.
// Thread t handles sorted slot t.
__global__ void d_markCellBounds(
    int* cellStart, int* cellEnd,
    const int* cellID, int nsites)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nsites) return;

    const int cur = cellID[t];
    if (t == 0 || cellID[t-1] != cur)
        cellStart[cur] = t;
    if (t == nsites-1 || cellID[t+1] != cur)
        cellEnd[cur] = t + 1;   // one-past-last
}


// Build neighbor list: for each group particle, search 3^Dim adjacent cells.
// d_neighborList[t*maxNeighbors + k] = global ID of k-th neighbor of particle siteList[t].
__global__ void d_buildNeighborList(
    int* neighborList, int* nNeighbors, int* overflowFlag,
    const int* siteList,
    const int* particleID, const int* cellStart, const int* cellEnd,
    const float* d_x,
    const float* d_L, const float* d_Lh,
    float rcut2, int maxNeighbors,
    int nCellsX, int nCellsY, int nCellsZ,
    float cellWidthX, float cellWidthY, float cellWidthZ,
    int Dim, int nsites)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nsites) return;

    const int id = siteList[t];
    const float* ri = d_x + id*Dim;

    // Home cell of this particle
    int cx = (int)(d_x[id*Dim + 0] / cellWidthX);
    cx = cx < 0 ? 0 : (cx >= nCellsX ? nCellsX-1 : cx);
    int cy = 0, cz = 0;
    if (Dim >= 2) {
        cy = (int)(d_x[id*Dim + 1] / cellWidthY);
        cy = cy < 0 ? 0 : (cy >= nCellsY ? nCellsY-1 : cy);
    }
    if (Dim >= 3) {
        cz = (int)(d_x[id*Dim + 2] / cellWidthZ);
        cz = cz < 0 ? 0 : (cz >= nCellsZ ? nCellsZ-1 : cz);
    }

    int count = 0;

    // Loop over 3^Dim neighboring cells (including self)
    int dzlo = (Dim >= 3) ? -1 : 0;
    int dzhi = (Dim >= 3) ?  1 : 0;
    int dylo = (Dim >= 2) ? -1 : 0;
    int dyhi = (Dim >= 2) ?  1 : 0;

    for (int dz = dzlo; dz <= dzhi; dz++) {
        int ncz = (cz + dz + nCellsZ) % nCellsZ;
        for (int dy = dylo; dy <= dyhi; dy++) {
            int ncy = (cy + dy + nCellsY) % nCellsY;
            for (int dx = -1; dx <= 1; dx++) {
                int ncx = (cx + dx + nCellsX) % nCellsX;

                int cellIdx = ncx + nCellsX * (ncy + nCellsY * ncz);
                int start = cellStart[cellIdx];
                int end   = cellEnd[cellIdx];
                if (start < 0) continue;

                for (int s = start; s < end; s++) {
                    int jd = particleID[s];
                    if (jd == id) continue;

                    float dr[3];
                    float dr2 = d_pbc_dr2f(dr, ri, d_x + jd*Dim, d_L, d_Lh, Dim);

                    if (dr2 < rcut2) {
                        if (count < maxNeighbors) {
                            neighborList[t * maxNeighbors + count] = jd;
                            count++;
                        } else {
                            atomicMax(overflowFlag, 1);
                        }
                    }
                }
            }
        }
    }

    nNeighbors[t] = count;
}


// ------------------------------------------------------------------

PS_NeighborList::PS_NeighborList(std::istringstream& iss, PS_Box* box)
    : mybox(box), maxNeighbors(64)
{

    iss >> grpName >> rcut;

    // Optional maxNeighbors override
    int tmp;
    iss >> tmp;
    if (!iss.fail()) maxNeighbors = tmp;

    rcut2 = rcut * rcut;
}

void PS_NeighborList::initializeNList() {
    groupInd = mybox->findGroupInteger(grpName);
    nsites   = mybox->psGroup[groupInd].nsites;

    // Cell dimensions: at least 1 cell per axis
    float Lx = mybox->Lh[0] * 2.0f;
    float Ly = (mybox->returnDimension() >= 2) ? mybox->Lh[1] * 2.0f : 1.0f;
    float Lz = (mybox->returnDimension() >= 3) ? mybox->Lh[2] * 2.0f : 1.0f;

    nCellsX = (int)(Lx / rcut); if (nCellsX < 1) nCellsX = 1;
    nCellsY = (mybox->returnDimension() >= 2) ? (int)(Ly / rcut) : 1;
    if (nCellsY < 1) nCellsY = 1;
    nCellsZ = (mybox->returnDimension() >= 3) ? (int)(Lz / rcut) : 1;
    if (nCellsZ < 1) nCellsZ = 1;
    nCells  = nCellsX * nCellsY * nCellsZ;

    cellWidthX = Lx / nCellsX;
    cellWidthY = Ly / nCellsY;
    cellWidthZ = Lz / nCellsZ;

    std::cout << "NeighborList: group=" << grpName
              << " rcut=" << rcut
              << " maxNeighbors=" << maxNeighbors
              << " cells=" << nCellsX << "x" << nCellsY << "x" << nCellsZ
              << std::endl;

    cudaMalloc(&d_cellID,           nsites * sizeof(int));
    cudaMalloc(&d_particleID,       nsites * sizeof(int));
    cudaMalloc(&d_cellID_sorted,    nsites * sizeof(int));
    cudaMalloc(&d_particleID_sorted,nsites * sizeof(int));
    cudaMalloc(&d_cellStart,        nCells * sizeof(int));
    cudaMalloc(&d_cellEnd,          nCells * sizeof(int));
    cudaMalloc(&d_neighborList,     nsites * maxNeighbors * sizeof(int));
    cudaMalloc(&d_nNeighbors,       nsites * sizeof(int));
}

PS_NeighborList::~PS_NeighborList() {
    cudaFree(d_cellID);
    cudaFree(d_particleID);
    cudaFree(d_cellID_sorted);
    cudaFree(d_particleID_sorted);
    cudaFree(d_cellStart);
    cudaFree(d_cellEnd);
    cudaFree(d_neighborList);
    cudaFree(d_nNeighbors);
}


void PS_NeighborList::build() {
    const int Dim    = mybox->returnDimension();
    const int nsBlk  = mybox->nsBlock;
    const int nsGrd  = (nsites + nsBlk - 1) / nsBlk;

    // 1. Assign each particle to a cell
    d_assignCellIDs<<<nsGrd, nsBlk>>>(
        d_cellID, d_particleID,
        mybox->psGroup[groupInd].d_siteList,
        mybox->d_x,
        cellWidthX, cellWidthY, cellWidthZ,
        nCellsX, nCellsY, nCellsZ,
        mybox->d_L, Dim, nsites);

    // 2. Sort particles by cell ID using CUB DeviceRadixSort (out-of-place).
    // CUB is used directly instead of Thrust to avoid execution-policy issues
    // that cause cudaErrorInvalidValue with some CUDA/WSL2 driver combinations.
    {
        void*  d_temp       = nullptr;
        size_t temp_bytes   = 0;
        // First call: query required temp-storage size
        cub::DeviceRadixSort::SortPairs(
            d_temp, temp_bytes,
            d_cellID, d_cellID_sorted,
            d_particleID, d_particleID_sorted,
            nsites);
        cudaMalloc(&d_temp, temp_bytes);
        // Second call: perform the sort
        cub::DeviceRadixSort::SortPairs(
            d_temp, temp_bytes,
            d_cellID, d_cellID_sorted,
            d_particleID, d_particleID_sorted,
            nsites);
        cudaFree(d_temp);
    }
    check_cudaError("PS_NeighborList::CUB sort");

    // 3. Mark cell boundaries (uses sorted arrays)
    cudaMemset(d_cellStart, -1, nCells * sizeof(int));
    cudaMemset(d_cellEnd,   -1, nCells * sizeof(int));
    d_markCellBounds<<<nsGrd, nsBlk>>>(d_cellStart, d_cellEnd, d_cellID_sorted, nsites);

    // 4. Build neighbor list
    cudaMemset(d_nNeighbors, 0, nsites * sizeof(int));

    int h_overflow = 0;
    int* d_overflow;
    cudaMalloc(&d_overflow, sizeof(int));
    cudaMemset(d_overflow, 0, sizeof(int));

    d_buildNeighborList<<<nsGrd, nsBlk>>>(
        d_neighborList, d_nNeighbors, d_overflow,
        mybox->psGroup[groupInd].d_siteList,
        d_particleID_sorted, d_cellStart, d_cellEnd,
        mybox->d_x, mybox->d_L, mybox->d_Lh,
        rcut2, maxNeighbors,
        nCellsX, nCellsY, nCellsZ,
        cellWidthX, cellWidthY, cellWidthZ,
        Dim, nsites);

    cudaMemcpy(&h_overflow, d_overflow, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_overflow);

    if (h_overflow)
        std::cout << "WARNING: PS_NeighborList: some particles exceeded maxNeighbors="
                  << maxNeighbors << "; increase maxNeighbors to avoid truncation." << std::endl;
}
