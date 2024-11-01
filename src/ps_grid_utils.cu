// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

//////////////////////////////////////////////
// July 2024                Rob Riggleman   //
// Utilities for mappings particles to the  //
// mesh grids for TILD-style simulations.   //
//////////////////////////////////////////////



// This routine determines the weights for spline interpolations from particle
// positions to density fields. This follows from the appendix of Deserno and Holm
// JCP V109 (1998) for particle-mesh-Ewald methods
// dx: distance to nearest grid point or dist to midpoint b/t grid points
//     this variable is (x-\bar{x})*H in Deserno and Holm's notation
__device__ void spline_get_weights(
    float dx,       // distance to nearest grid point or midpoint
    float H,        // Grid spacing in this direction
    float* W,       // [pmeorder+1] vector to store the weights
    int pmeorder    // order of the interpolation scheme
    ) {

    float sx = dx / H;

    float sx2, sx3, sx4;

    if (pmeorder == 0)
        W[0] = 1.f ;

    else if (pmeorder == 1) {
        W[0] = (0.5f - sx);
        W[1] = (0.5f + sx);
    }

    else if (pmeorder == 2) {
        sx2 = sx * sx;

        W[0] = (0.125f - 0.5f * sx + 0.5f * sx2);
        W[1] = (0.75f - sx2);
        W[2] = (0.125f + 0.5f * sx + 0.5f * sx2);

    }

    else if (pmeorder == 3) {
        sx2 = sx * sx;
        sx3 = sx2 * sx;

        W[0] = (1.f - 6.f * sx + 12.f * sx2 - 8.f * sx3) / 48.f;
        W[1] = (23.f - 30.f * sx - 12.f * sx2 + 24.f * sx3) / 48.f;
        W[2] = (23.f + 30.f * sx - 12.f * sx2 - 24.f * sx3) / 48.f;
        W[3] = (1.f + 6.f * sx + 12.f * sx2 + 8.f * sx3) / 48.f;
    }

    else if (pmeorder == 4) {
        sx2 = sx * sx;
        sx3 = sx2 * sx2;
        sx4 = sx2 * sx2;

        W[0] = (1.f - 8.f * sx + 24.f * sx2 - 32.f * sx3 + 16.f * sx4) / 384.f;
        W[1] = (19.f - 44.f * sx + 24.f * sx2 + 16.f * sx3 - 16.f * sx4) / 96.f;
        W[2] = (115.f - 120.f * sx2 + 48.f * sx4) / 192.f;
        W[3] = (19.f + 44.f * sx + 24.f * sx2 - 16.f * sx3 - 16.f * sx4) / 96.f;
        W[4] = (1.f + 8.f * sx + 24.f * sx2 + 32.f * sx3 + 16.f * sx4) / 384.f;
    }

}


// Creates of the list of grid indices \in [0,M) that a particle maps to
// and generates the weights for those grid points. 
// Can handle 1- and 2-D calculations.
// Each particle contributes to (pmeorder+1)^Dim grid points
__global__ void d_calcGridWeights(
    float* d_grid_W,    // [ns*grid_per_partic] weights for particle on each grid point
    int* d_grid_inds,   // [ns*grid_per_partic] grid indices for each particle
    const float* d_x,   // [ns*Dim] particle positions
    const int* d_Nx,    // [Dim] number of grid points in each direction
    const float* d_dx,  // [Dim] grid spacing in each direction
    const int ns,       // Numberr of particles in the box
    const int pmeorder, // spline interpolation order
    const int M,        // Total number of grid poitns, M= \prod_i Nx[i]
    const int Dim       // System dimensionality
    ) {

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ns)
        return;

    float W[3][6];
    float gdx, W3, gvol;
    int g_ind[3];
    int nn[3];
    int grid_ct = 0;
    int Mindex, grid_per_partic = 1;

    gvol = 1.0f;
    for (int j = 0; j < Dim; j++) {
        gvol *= d_dx[j];
        grid_per_partic *= (pmeorder + 1);
    }

    int INDMAX = ns * grid_per_partic;

    for (int j = 0; j < Dim; j++) {
        if (pmeorder % 2 == 0) {
            g_ind[j] = int((d_x[id * Dim + j] + 0.5f * d_dx[j]) / d_dx[j]);
            gdx = d_x[id * Dim + j] - float(g_ind[j]) * d_dx[j];
        }
        else {
            g_ind[j] = int(d_x[id * Dim + j] / d_dx[j]);
            gdx = d_x[id * Dim + j] - (float(g_ind[j]) + 0.5f) * d_dx[j];
        }
        spline_get_weights(gdx, d_dx[j], W[j], pmeorder);
    }

    // 
    if (Dim == 2) {
        int order_shift = pmeorder / 2;

        for (int ix = 0; ix < pmeorder + 1; ix++) {

            nn[0] = (g_ind[0] + ix - order_shift + d_Nx[0]) % d_Nx[0];

            // nn[0] = g_ind[0] + ix - order_shift;

            // while (nn[0] < 0) nn[0] += d_Nx[0];
            // while (nn[0] >= d_Nx[0]) nn[0] -= d_Nx[0];

            for (int iy = 0; iy < pmeorder + 1; iy++) {

                nn[1] = (g_ind[1] + iy - order_shift + d_Nx[1]) % d_Nx[1];

                // nn[1] = g_ind[1] + iy - order_shift;

                // while (nn[1] < 0) nn[1] += d_Nx[1];
                // while (nn[1] >= d_Nx[1]) nn[1] -= d_Nx[1];

                Mindex = nn[1] * d_Nx[0] + nn[0];

                W3 = W[0][ix] * W[1][iy] / gvol;
                
                int NSIND = id * grid_per_partic + grid_ct;

                if ( NSIND < INDMAX ) {
                    d_grid_inds[NSIND] = Mindex;
                    d_grid_W[NSIND] = W3;

                    grid_ct++;  
                }
                

            }// iy = 0:pmeorder+1
        }// ix=0:pmeorder+1
    }// if Dim==2

    else if (Dim == 3) {
        int order_shift = pmeorder / 2;

        for (int ix = 0; ix < pmeorder + 1; ix++) {

            nn[0] = (g_ind[0] + ix - order_shift + d_Nx[0]) % d_Nx[0];
            // nn[0] = g_ind[0] + ix - order_shift;

            // while (nn[0] < 0) nn[0] += d_Nx[0];
            // while (nn[0] >= d_Nx[0]) nn[0] -= d_Nx[0];

            for (int iy = 0; iy < pmeorder + 1; iy++) {

                nn[1] = (g_ind[1] + iy - order_shift + d_Nx[1]) % d_Nx[1];

                // nn[1] = g_ind[1] + iy - order_shift;

                // while (nn[1] < 0) nn[1] += d_Nx[1];
                // while (nn[1] >= d_Nx[1]) nn[1] -= d_Nx[1];

                for (int iz = 0; iz < pmeorder + 1; iz++) {

                    nn[2] = (g_ind[2] + iz - order_shift + d_Nx[2]) % d_Nx[2];

                    // nn[2] = g_ind[2] + iz - order_shift;

                    // while (nn[2] < 0) nn[2] += d_Nx[2];
                    // while (nn[2] >= d_Nx[2]) nn[2] -= d_Nx[2];

                    Mindex = nn[0] + (nn[1] + nn[2] * d_Nx[1]) * d_Nx[0];

                    W3 = W[0][ix] * W[1][iy] * W[2][iz] / gvol;

                    int NSIND = id * grid_per_partic + grid_ct;
                    if ( NSIND < INDMAX ) {
                        d_grid_inds[NSIND] = Mindex;
                        d_grid_W[NSIND] = W3;

                        grid_ct++;
                    }
                    
                }// iz=0:pmeorder+1
            }// iy=0:pmeorder+1
        }// ix=0:pmeorder+1

    }// if Dim == 3

} // d_calcGridWeights



// __global__ void d_charge_grid_charges(float* d_x, float* d_grid_W, int* d_grid_inds,
//     int* d_tp, float* d_rho, // d_rho, Has dimensions ntypes*M
//     const int* d_Nx, const float* d_dx, const float V,
//     const int ns, const int pmeorder, const int M, const int Dim,
//     float* d_charge_density, float* charges, int charge_flag) {// float* d_charges_over_M) {

//     const int id = blockIdx.x * blockDim.x + threadIdx.x;
//     if (id >= ns)
//         return;

//     float W[3][6];
//     float gdx, W3, gvol, W3_charges;
//     int g_ind[3];
//     int nn[3];
//     int grid_ct = 0;
//     int Mindex, rho_ind, grid_per_partic = 1;
//     int id_typ = d_tp[id];

//     gvol = 1.0f;
//     for (int j = 0; j < Dim; j++) {
//         gvol *= d_dx[j];
//         grid_per_partic *= (pmeorder + 1);
//     }


//     for (int j = 0; j < Dim; j++) {
//         if (pmeorder % 2 == 0) {
//             g_ind[j] = int((d_x[id * Dim + j] + 0.5f * d_dx[j]) / d_dx[j]);
//             gdx = d_x[id * Dim + j] - float(g_ind[j]) * d_dx[j];
//         }
//         else {
//             g_ind[j] = int(d_x[id * Dim + j] / d_dx[j]);
//             gdx = d_x[id * Dim + j] - (float(g_ind[j]) + 0.5f) * d_dx[j];
//         }
//         spline_get_weights(gdx, d_dx[j], W[j], pmeorder);
//     }


//     // 
//     if (Dim == 2) {
//         int order_shift = pmeorder / 2;

//         for (int ix = 0; ix < pmeorder + 1; ix++) {

//             nn[0] = g_ind[0] + ix - order_shift;

//             if (nn[0] < 0) nn[0] += d_Nx[0];
//             else if (nn[0] >= d_Nx[0]) nn[0] -= d_Nx[0];

//             for (int iy = 0; iy < pmeorder + 1; iy++) {

//                 nn[1] = g_ind[1] + iy - order_shift;

//                 if (nn[1] < 0) nn[1] += d_Nx[1];
//                 else if (nn[1] >= d_Nx[1]) nn[1] -= d_Nx[1];

//                 Mindex = nn[1] * d_Nx[0] + nn[0];
//                 rho_ind = id_typ * M + Mindex;

//                 W3 = W[0][ix] * W[1][iy] / gvol;
//                 W3_charges = W[0][ix] * W[1][iy] / gvol;

//                 atomicAdd(&d_rho[rho_ind], W3);

//                 atomicAdd(&d_charge_density[Mindex], W3_charges * charges[id]);

//                 d_grid_inds[id * grid_per_partic + grid_ct] = Mindex;
//                 d_grid_W[id * grid_per_partic + grid_ct] = W3;

//                 grid_ct++;

//             }// iy = 0:pmeorder+1
//         }// ix=0:pmeorder+1
//     }// if Dim==2

//     else if (Dim == 3) {
//         int order_shift = pmeorder / 2;

//         for (int ix = 0; ix < pmeorder + 1; ix++) {

//             nn[0] = g_ind[0] + ix - order_shift;

//             if (nn[0] < 0) nn[0] += d_Nx[0];
//             else if (nn[0] >= d_Nx[0]) nn[0] -= d_Nx[0];

//             for (int iy = 0; iy < pmeorder + 1; iy++) {

//                 nn[1] = g_ind[1] + iy - order_shift;

//                 nn[1] = nn[1]%d_Nx[1];
//                 // if (nn[1] < 0) nn[1] += d_Nx[1];
//                 // else if (nn[1] >= d_Nx[1]) nn[1] -= d_Nx[1];

//                 for (int iz = 0; iz < pmeorder + 1; iz++) {

//                     nn[2] = g_ind[2] + iz - order_shift;

//                     if (nn[2] < 0) nn[2] += d_Nx[2];
//                     else if (nn[2] >= d_Nx[2]) nn[2] -= d_Nx[2];

//                     Mindex = nn[0] + (nn[1] + nn[2] * d_Nx[1]) * d_Nx[0];
//                     rho_ind = id_typ * M + Mindex;

//                     W3 = W[0][ix] * W[1][iy] * W[2][iz] / gvol;
//                     W3_charges = W[0][ix] * W[1][iy] * W[2][iz] / gvol;

//                     atomicAdd(&d_rho[rho_ind], W3);

//                     atomicAdd(&d_charge_density[Mindex], W3_charges * charges[id]);

//                     d_grid_inds[id * grid_per_partic + grid_ct] = Mindex;
//                     d_grid_W[id * grid_per_partic + grid_ct] = W3;

//                     grid_ct++;
//                 }
//             }
//         }// for ix
//     }// if Dim == 3

// }


// __global__ void d_charge_grid(float* d_x, float* d_grid_W, int* d_grid_inds,
//     int* d_tp, float* d_rho, // d_rho, Has dimensions ntypes*M
//     const int* d_Nx, const float* d_dx, const float V,
//     const int ns, const int pmeorder, const int M, const int Dim) {

//     const int id = blockIdx.x * blockDim.x + threadIdx.x;
//     if (id >= ns)
//         return;

//     float W[3][6];
//     float gdx, W3, gvol;
//     int g_ind[3];
//     int nn[3];
//     int grid_ct = 0;
//     int Mindex, rho_ind, grid_per_partic = 1;
//     int id_typ = d_tp[id];

//     gvol = 1.0f;
//     for (int j = 0; j < Dim; j++) {
//         gvol *= d_dx[j];
//         grid_per_partic *= (pmeorder + 1);
//     }


//     for (int j = 0; j < Dim; j++) {
//         if (pmeorder % 2 == 0) {
//             g_ind[j] = int((d_x[id * Dim + j] + 0.5f * d_dx[j]) / d_dx[j]);
//             gdx = d_x[id * Dim + j] - float(g_ind[j]) * d_dx[j];
//         }
//         else {
//             g_ind[j] = int(d_x[id * Dim + j] / d_dx[j]);
//             gdx = d_x[id * Dim + j] - (float(g_ind[j]) + 0.5f) * d_dx[j];
//         }
//         spline_get_weights(gdx, d_dx[j], W[j], pmeorder);
//     }


//     // 
//     if (Dim == 2) {
//         int order_shift = pmeorder / 2;

//         for (int ix = 0; ix < pmeorder + 1; ix++) {

//             nn[0] = g_ind[0] + ix - order_shift;

//             if (nn[0] < 0) nn[0] += d_Nx[0];
//             else if (nn[0] >= d_Nx[0]) nn[0] -= d_Nx[0];

//             for (int iy = 0; iy < pmeorder + 1; iy++) {

//                 nn[1] = g_ind[1] + iy - order_shift;

//                 if (nn[1] < 0) nn[1] += d_Nx[1];
//                 else if (nn[1] >= d_Nx[1]) nn[1] -= d_Nx[1];

//                 Mindex = nn[1] * d_Nx[0] + nn[0];
//                 rho_ind = id_typ * M + Mindex;

//                 W3 = W[0][ix] * W[1][iy] / gvol;

//                 atomicAdd(&d_rho[rho_ind], W3);
//                 d_grid_inds[id * grid_per_partic + grid_ct] = Mindex;
//                 d_grid_W[id * grid_per_partic + grid_ct] = W3;

//                 grid_ct++;

//             }// iy = 0:pmeorder+1
//         }// ix=0:pmeorder+1
//     }// if Dim==2

//     else if (Dim == 3) {
//         int order_shift = pmeorder / 2;

//         for (int ix = 0; ix < pmeorder + 1; ix++) {

//             nn[0] = g_ind[0] + ix - order_shift;

//             if (nn[0] < 0) nn[0] += d_Nx[0];
//             else if (nn[0] >= d_Nx[0]) nn[0] -= d_Nx[0];

//             for (int iy = 0; iy < pmeorder + 1; iy++) {

//                 nn[1] = g_ind[1] + iy - order_shift;

//                 if (nn[1] < 0) nn[1] += d_Nx[1];
//                 else if (nn[1] >= d_Nx[1]) nn[1] -= d_Nx[1];

//                 for (int iz = 0; iz < pmeorder + 1; iz++) {

//                     nn[2] = g_ind[2] + iz - order_shift;

//                     if (nn[2] < 0) nn[2] += d_Nx[2];
//                     else if (nn[2] >= d_Nx[2]) nn[2] -= d_Nx[2];

//                     Mindex = nn[0] + (nn[1] + nn[2] * d_Nx[1]) * d_Nx[0];
//                     rho_ind = id_typ * M + Mindex;

//                     W3 = W[0][ix] * W[1][iy] * W[2][iz] / gvol;

//                     atomicAdd(&d_rho[rho_ind], W3);

//                     d_grid_inds[id * grid_per_partic + grid_ct] = Mindex;
//                     d_grid_W[id * grid_per_partic + grid_ct] = W3;

//                     grid_ct++;
//                 }
//             }
//         }// for ix

//     }// if Dim == 3

// }


__global__ void d_add_grid_forces_charges_2D(float* fp, const float* fgx,
    const float* fgy, const float* d_t_charge_density_field, const float* d_grid_W,
    const int* d_grid_inds, const float gvol, const float* charges,
    const int grid_per_partic, const int ns, const int M, const int Dim) {

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ns)
        return;

    int gind;
    float W3;

    for (int m = 0; m < grid_per_partic; m++) {
        gind = d_grid_inds[id * grid_per_partic + m];

        W3 = d_grid_W[id * grid_per_partic + m];

        fp[id * Dim + 0] += fgx[gind] * W3 * gvol * charges[id];
        fp[id * Dim + 1] += fgy[gind] * W3 * gvol * charges[id];
        
    }

}

__global__ void d_add_grid_forces2D(float* fp, const float* fgx,
    const float* fgy, const float* d_t_rho, const float* d_grid_W, 
    const int* d_grid_inds, const int* d_tp, const float gvol, 
    const int grid_per_partic, const int ns, const int M, const int Dim ) {

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ns)
        return;

    int id_typ = d_tp[id];
    int gind, typ_ind;
    float W3;

    for (int m = 0; m < grid_per_partic; m++) {
        gind = d_grid_inds[id * grid_per_partic + m];

        W3 = d_grid_W[id * grid_per_partic + m];

        typ_ind = id_typ * M + gind;

        if (d_t_rho[typ_ind] > 0.f) {
            fp[id * Dim + 0] += fgx[typ_ind] * W3 * gvol / d_t_rho[typ_ind];
            fp[id * Dim + 1] += fgy[typ_ind] * W3 * gvol / d_t_rho[typ_ind];
        }
        
    }
    
}

__global__ void d_add_grid_forces_charges_3D(float* fp, const float* fgx,
    const float* fgy, const float* fgz, const float* d_t_charge_density_field, const float* d_grid_W,
    const int* d_grid_inds, const float gvol, const float* charges,
    const int grid_per_partic, const int ns, const int M, const int Dim) {

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ns)
        return;

    int gind;
    float W3;

    for (int m = 0; m < grid_per_partic; m++) {
        gind = d_grid_inds[id * grid_per_partic + m];

        W3 = d_grid_W[id * grid_per_partic + m];

        fp[id * Dim + 0] += fgx[gind] * W3 * gvol * charges[id];
        fp[id * Dim + 1] += fgy[gind] * W3 * gvol * charges[id];
        fp[id * Dim + 2] += fgz[gind] * W3 * gvol * charges[id];
        
    }

}

__global__ void d_add_grid_forces3D(float* fp, const float* fgx,
    const float* fgy, const float* fgz, const float* d_t_rho, const float* d_grid_W,
    const int* d_grid_inds, const int* d_tp, const float gvol,
    const int grid_per_partic, const int ns, const int M, const int Dim) {

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ns)
        return;

    int id_typ = d_tp[id];
    int gind, typ_ind;
    float W3;

    for (int m = 0; m < grid_per_partic; m++) {
        gind = d_grid_inds[id * grid_per_partic + m];

        W3 = d_grid_W[id * grid_per_partic + m];

        typ_ind = id_typ * M + gind;

        if (d_t_rho[typ_ind] > 0.f) {
            fp[id * Dim + 0] += fgx[typ_ind] * W3 * gvol / d_t_rho[typ_ind];
            fp[id * Dim + 1] += fgy[typ_ind] * W3 * gvol / d_t_rho[typ_ind];
            fp[id * Dim + 2] += fgz[typ_ind] * W3 * gvol / d_t_rho[typ_ind];
        }

    }

}
