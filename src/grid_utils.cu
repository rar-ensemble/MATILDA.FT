// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "potential_charges.h"


__global__ void d_charge_grid_charges(float*, float*, int*, int*, float*, const int*,
    const float*, const float, const int, const int, const int, const int, float*, float*, int);
__global__ void d_charge_grid(float*, float*, int*, int*, float*, const int*,
    const float*, const float, const int, const int, const int, const int);
__global__ void d_zero_all_ntyp(float*, int, int);
__global__ void d_prep_components(float*, float*, float*,
	const int, const int, const int);
__global__ void d_zero_float_vector(float*, int);

void prepareDensityFields() {


    // Zeros the types*M density field
    d_zero_all_ntyp<<<M_Grid, M_Block>>>(d_all_rho, M, ntypes);


    if ( Charges::do_charges == 1 ) {
        d_zero_float_vector<<<M_Grid, M_Block>>>(d_charge_density_field, M);
    check_cudaError("d_charge_density_field");
		d_zero_float_vector<<<M_Grid, M_Block>>>(d_electrostatic_potential, M);
    check_cudaError("d_electrostatic_potential");
    }
    

    // Fills the ntypes*M density field
    // and Fills d_charge_density_field if charges is flagged
    if (Charges::do_charges == 1) {
        d_charge_grid_charges<<<ns_Grid, ns_Block>>>(d_x, d_grid_W,
            d_grid_inds, d_typ, d_all_rho, d_Nx, d_dx,
            V, ns, pmeorder, M, Dim, d_charge_density_field, d_charges, Charges::do_charges);
        
    }
    else {
        d_charge_grid<<<ns_Grid, ns_Block>>>(d_x, d_grid_W,
            d_grid_inds, d_typ, d_all_rho, d_Nx, d_dx,
            V, ns, pmeorder, M, Dim);
    }

    check_cudaError("d_charge_grid");


    // Zeros the grid forces and copies the density into
    // its class structure.
    for (int i = 0; i < ntypes; i++) {
        d_prep_components<<<M_Grid, M_Block>>>
            (Components[i].d_force, Components[i].d_rho, d_all_rho, i, M, Dim);
    }

    check_cudaError("d_prep_components");

}
