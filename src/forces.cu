// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "timing.h"

__global__ void d_zero_all_ntyp(float*, int, int);
__global__ void d_zero_all_directions_charges(float*, int);
__global__ void d_bonds(int*, int*, int*, float*,
	float*, float*, float*, float*, float*, int, int, int, int*);
__global__ void d_angles(const float*, float*, const float*,
    const float*, const int*, const int*, const int*, const int*, const int*,
    const int*, const float*, const float*, const int, const int,
    const int);


__global__ void d_zero_particle_forces(float*, int, int);
__global__ void d_real2complex(float*, cufftComplex*, int M);
__global__ void d_add_grid_forces2D(float*, const float*,
	const float*, const float*, const float*,
	const int*, const int*, const float,
	const int, const int, const int, const int);
__global__ void d_add_grid_forces_charges_2D(float*, const float*,
	const float*, const float*, const float*,
	const int*, const float, const float*,
	const int, const int, const int, const int);
__global__ void d_add_grid_forces3D(float*, const float*,
	const float*, const float*, const float*, const float*,
	const int*, const int*, const float,
	const int, const int, const int, const int);
__global__ void d_add_grid_forces_charges_3D(float*, const float*,
	const float*, const float*, const float*, const float*,
	const int*, const float, const float*,
	const int, const int, const int, const int);

void cuda_collect_rho(void);
void write_grid_data(const char*, float*);
void cuda_collect_x(void);
__global__ void d_make_dens_step(float*, float*, float*, int*, int, int, int);
void prepareDensityFields(void);

void forces() {

    prepareDensityFields();

	d_zero_all_ntyp<<<M_Grid, M_Block>>>(d_all_fx, M, ntypes);
	d_zero_all_ntyp<<<M_Grid, M_Block>>>(d_all_fy, M, ntypes);
	if ( Dim == 3 ) d_zero_all_ntyp<<<M_Grid, M_Block>>>(d_all_fz, M, ntypes);

	if (Charges::do_charges == 1) {

		d_zero_all_directions_charges<<<M_Grid, M_Block>>>(d_all_fx_charges, M);
		d_zero_all_directions_charges<<<M_Grid, M_Block>>>(d_all_fy_charges, M);

		if (Dim == 3) d_zero_all_directions_charges<<<M_Grid, M_Block>>>(d_all_fz_charges, M);
	}

	check_cudaError("forces zeroed");

	// Zeros the forces on the particles
	d_zero_particle_forces<<<ns_Grid, ns_Block>>>(d_f, ns, Dim);

	for (auto Iter: Potentials){
		Iter->CalcForces();
	}



	check_cudaError("d_zero_particle_forces");

	// Accumulates forces from grid onto particles
	if (Dim == 2) {
		if (Charges::do_charges == 1) {
			d_add_grid_forces_charges_2D<<<ns_Grid, ns_Block>>>(d_f, d_all_fx_charges,
				d_all_fy_charges, d_charge_density_field, d_grid_W, d_grid_inds, gvol,
				d_charges, grid_per_partic, ns, M, Dim);
		}
		
		d_add_grid_forces2D<<<ns_Grid, ns_Block>>>(d_f, d_all_fx,
			d_all_fy, d_all_rho, d_grid_W, d_grid_inds, d_typ, gvol,
			grid_per_partic, ns, M, Dim);
		
	}

	else if (Dim == 3) {
		if (Charges::do_charges == 1) {
			
			d_add_grid_forces_charges_3D<<<ns_Grid, ns_Block>>>(d_f, d_all_fx_charges,
				d_all_fy_charges, d_all_fz_charges, d_charge_density_field, d_grid_W, d_grid_inds, gvol,
				d_charges, grid_per_partic, ns, M, Dim);
		}
		
		d_add_grid_forces3D<<<ns_Grid, ns_Block>>>(d_f, d_all_fx,
			d_all_fy, d_all_fz, d_all_rho, d_grid_W, d_grid_inds, d_typ, gvol,
			grid_per_partic, ns, M, Dim);
		
	}


	check_cudaError("d_add_grid_forces");


	// Accumulates bonded forces
	if (n_total_bonds > 0) {
		d_bonds<<<ns_Grid, ns_Block>>>(d_n_bonds, d_bonded_to,
			d_bond_type, d_bond_req, d_bond_k, d_x, d_f,
			d_L, d_Lh, ns, MAX_BONDS, Dim, d_bond_style);

    check_cudaError("bond forces");
	}

    if ( n_total_angles > 0 ) {
        d_angles<<<ns_Grid, ns_Block>>>(d_x, d_f, d_angle_k,
            d_angle_theta_eq, d_angleIntStyle, d_n_angles, d_angle_type,
            d_angle_first,d_angle_mid, d_angle_end, d_L, d_Lh,
            ns, MAX_ANGLES, Dim);
        
        check_cudaError("angle forces");
    }



}
