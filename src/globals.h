// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include <iostream>
#include <fstream>
#include <string>
#include <stdio.h>
#include <complex>
#include <vector>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include "field_component.h"
#include "potential_gaussian.h"
#include "potential_fieldphases.h"
#include "potential_erf.h"
#include "potential_gaussian_erf.h"
#include "potential_charges.h"
#include "tensor_potential_MaierSaupe.h"
#include "group.h"
#include "group_type.h"
#include "integrator.h"
#include "nlist.h"
#include "Compute.h"
#include "Compute_widom.h"
#include "Compute_avg_sk.h"
#include "Compute_sk.h"
#include "Compute_chempot.h"
#include "Extraforce.h"
#include "Extraforce_langevin.h"
#include "Extraforce_midpush.h"
#include "global_templated_functions.h"
#include "group_region.h"
#include "include_libs.h"

#include "Box.h"

#include <thrust/sequence.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/device_ptr.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>

#define PI 3.141592654f
#define PI2 6.2831853071f
#define PI4 12.56637061435917295384f

#ifndef MAIN
extern
#endif
cudaStream_t stream1, stream2, stream3;


/// Variables for s(k) calculations
#ifndef MAIN
extern
#endif
float ** avg_sk;

#ifndef MAIN
extern
#endif
int n_avg_calc;

#ifndef MAIN
extern
#endif
int ns, Dim, ntypes, step, max_steps, * tp, * molecID, grid_freq, Ng, * bond_style,
traj_freq, log_freq, bond_log_freq, struc_freq, bin_freq, equil_grid_freq, 
equil_traj_freq, equil_log_freq, equil_struc_freq, equil_bin_freq,
prod_grid_freq, prod_traj_freq, prod_log_freq, prod_struc_freq, prod_bin_freq, skip_steps, mem_use, device_mem_use, RAND_SEED, read_rand_seed,
Nx[3], pmeorder, M, grid_per_partic,
n_total_bonds, n_total_angles, nbond_types, nangle_types, *angleIntStyle,
* n_bonds, * n_angles, ** bonded_to, ** bond_type,
** angle_first, ** angle_mid, ** angle_end, ** angle_type,
threads, n_P_comps,
MAX_BONDS, MAX_ANGLES,
extra_ns_memory,    // Allocate memory for extra sites (e.g., ghost molecule Widom method)
n_molecules,        // Total number of molecules
prod_steps,         // Number of steps to run
equil_steps,        // Number of steps to equilibrate, default 0
equilData,         // Flag to write equilibration data
allocate_velocities,// Flag to allocate particle velocities
gsd_freq,           // Frequency to write GSD file
global_step,
LOW_DENS_FLAG,
traj_skip,
pos_skip,
grid_skip,


n_groups,           // Total number of group definitions.
n_integrators,      // total number of integrators
using_GJF,          // Flag to know to allocate GJF memory
n_neighbor_lists,   // Total number of neighbor lists allocated
n_computes,         // Total number of computes
n_extra_forces,     // Number of ''extraforce'' routines
n_MaierSaupe,       // Number of Maier-Saupe pair styles
particle_sim,       // Flag for whether we're doing a particle simulation
field_sim,          // Flag for whether we're doing a field-based simulation
do_charges,
GFLAG,
GRID_UPDATE_FREQ,
*GRID_STATE;


#ifndef MAIN
extern
#endif
cufftComplex **d_calculated_rho_all;

#ifndef MAIN
extern
#endif
std::vector<unsigned int> list_of_bond_type, 
 list_of_bond_partners,
 list_of_angle_type,
 list_of_angle_partners; 

#ifndef MAIN
extern
#endif
float L[6], Lh[6], V, ** x, ** xo, ** f, ** v, * mass, * Diff, * h_ns_float,
delt, * bond_k, * bond_req, Ubond, Udynamicbond, * angle_k, * angle_theta_eq, Uangle,
dx[3], * tmp, * tmp2, * all_rho, gvol, noise_mag,
Upe, Unb, * Ptens, * partic_bondE, * partic_bondVir, * bondVir, *angleVir,
// 2D: Ptens[0] = xx, Ptens[1] = yy, Ptens[2] = xy
// 3D: Ptens[0] = xx, Ptens[1] = yy, Ptens[2] = zz, [3] = xy, [4] = xz, [5] = yz
* molec_Ubond,                  // Bonded energy for each molecule
* charges, charge_bjerrum_length, charge_smearing_length,
* charge_density_field, * electrostatic_energy, 
* electrostatic_potential, * electric_field,
* electrostatic_energy_direct_computation;

#ifndef MAIN
extern
#endif
bool equil;         // Flag to know if equilibration is to be done

#ifndef MAIN
extern
#endif
std::string dump_name, equil_name, input_file, gsd_name;

#ifndef MAIN
extern
#endif
std::vector<std::string> angleStyle;

#ifndef MAIN
extern
#endif
std::complex<float> I, * k_tmp ;


#ifndef MAIN
extern
#endif
float* d_x, * d_v, * d_f, * d_L, * d_Lh,
* d_bond_req, * d_bond_k, * d_3n_tmp,
* d_tmp, * d_tmp2, * d_dx, * d_all_rho,
* d_grid_W, * d_all_fx, * d_all_fy, * d_all_fz,
* d_all_fx_charges, * d_all_fy_charges, * d_all_fz_charges,
* d_bondE, * d_bondVir,
* d_xo, * d_prev_noise, * d_mass, * d_Diff,
* d_charges, * d_charge_density_field, * d_electrostatic_potential, * d_electric_field,
* d_angle_k, * d_angle_theta_eq ;

#ifndef MAIN
extern
#endif
cufftComplex* d_cpx1, * d_cpx2, * cpx1, * cpx2, *d_cpxx, *d_cpxy, *d_cpxz;

#ifndef MAIN
extern
#endif
bool* d_nan;

#ifndef MAIN
extern
#endif
cufftHandle fftplan;

#ifndef MAIN
extern
#endif
int d_ns, * d_typ, * d_Nx, ns_Block, ns_Grid, M_Block, M_Grid, 
* d_grid_inds, * d_molecID,
* d_n_bonds, * d_n_angles, * d_bonded_to, * d_bond_type, *d_bond_style,
* d_angle_first, * d_angle_mid, * d_angle_end, *d_angle_type, *d_angleIntStyle ;

#ifndef MAIN
extern
#endif
cudaError_t cudaReturn;

#ifndef MAIN
extern
#endif
FieldComponent* Components;


#ifndef MAIN
extern
#endif
std::vector<FieldPhase> Fields;

#ifndef MAIN
extern
#endif
std::vector<Erf> Erfs;

#ifndef MAIN
extern
#endif
std::vector<ExtraForce*> ExtraForces;

#ifndef MAIN
extern
#endif
std::vector<ExtraForce*> DynamicBonds;

#ifndef MAIN
extern
#endif
std::vector<Group*> Groups;


#ifndef MAIN
extern
#endif
std::vector<Box*> box;

#ifndef MAIN
extern
#endif
std::vector<NList*> NLists;

#ifndef MAIN
extern
#endif
std::vector<Integrator*> Integrators;

#ifndef MAIN
extern
#endif
std::vector<Potential*> Potentials;

#ifndef MAIN
extern
#endif
Charges* MasterCharge;

#ifndef MAIN
extern
#endif
std::vector<Compute*> Computes;

#ifndef MAIN
extern
#endif
std::vector<float> dr_Triggers;

#ifndef MAIN
extern
#endif
float MAX_DISP;

#ifndef MAIN
extern
#endif
curandState* d_states;

#ifndef MAIN
extern
#endif
float U_Electro_old;


void die(const char*);
void check_cudaError(const char*);
float pbc_mdr2(float*, float*, float*);
float get_k(int, float*, int);
void get_r(int, float*);
float integ_trapPBC(float*);
void write_grid_data(const char*, float*);
