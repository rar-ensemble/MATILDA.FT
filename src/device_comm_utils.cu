// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
__global__ void d_copyPositions(float*, float*, int, int);

void update_device_positions(float** out, float* d_target) {
    int i, j;
	for (i = 0; i < ns; i++) {
		for (j = 0; j < Dim; j++) {
			h_ns_float[i * Dim + j] = out[i][j];
		}
	}
	cudaMemcpy(d_target, h_ns_float, ns * Dim * sizeof(float),
		cudaMemcpyHostToDevice);
}

void send_3n_to_device(float** out, float *d_target) {
	int i, j;
	for (i = 0; i < ns; i++) {
		for (j = 0; j < Dim; j++) {
			h_ns_float[i * Dim + j] = out[i][j];
		}
	}
	cudaMemcpy(d_target, h_ns_float, ns * Dim * sizeof(float),
		cudaMemcpyHostToDevice);

	if ( Integrator::using_GJF )
		d_copyPositions<<<ns_Grid, ns_Block>>>(d_xo, d_x, Dim, ns);
}


void cuda_collect_x() {
	cudaMemcpy(h_ns_float, d_x, ns * Dim * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < ns; i++)
		for (int j = 0; j < Dim; j++)
			x[i][j] = h_ns_float[i * Dim + j];

	if ( Charges::do_charges == 1 )
		cudaMemcpy(charges, d_charges, ns * sizeof(float), cudaMemcpyDeviceToHost);
}

void cuda_collect_rho() {
	cudaMemcpy(all_rho, d_all_rho, ntypes * M * sizeof(float), 
		cudaMemcpyDeviceToHost);
	// Copys all_rho to device_all_rho

	for (int i = 0; i < ntypes; i++) {
		for (int j = 0; j < M; j++) {
			Components[i].rho[j] = all_rho[i * M + j];
		}
	}
}



void cuda_collect_charge_density_field() {

	cudaMemcpy(charge_density_field, d_charge_density_field, M * sizeof(float),
		cudaMemcpyDeviceToHost);

	check_cudaError("cuda_collect_charge_density_field");
}

void cuda_collect_electric_field() {

	cudaMemcpy(electric_field, d_electric_field, M * Dim * sizeof(float),
		cudaMemcpyDeviceToHost);

	check_cudaError("cuda_collect_electric_field");
}


void cuda_collect_electrostatic_potential() {

	cudaMemcpy(electrostatic_potential, d_electrostatic_potential, M * sizeof(float),
		cudaMemcpyDeviceToHost);

	check_cudaError("cuda_collect_electrostatic_potential");
}


void cuda_collect_f() {
	cudaMemcpy(h_ns_float, d_f, ns * Dim * sizeof(float), cudaMemcpyDeviceToHost);
	for (int i = 0; i < ns; i++)
		for (int j = 0; j < Dim; j++)
			f[i][j] = h_ns_float[i * Dim + j];
}


void send_box_params_to_device() {

	// Box geometry //
	cudaMemcpy(d_L, L, 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Lh, Lh, 3 * sizeof(float), cudaMemcpyHostToDevice);

    int nsSend = ns + extra_ns_memory;
	cudaMemcpy(d_typ, tp, nsSend * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_molecID, molecID, nsSend * sizeof(int), cudaMemcpyHostToDevice);

	// Bonding information //
	cudaMemcpy(d_n_bonds, n_bonds, nsSend * sizeof(int), cudaMemcpyHostToDevice);
	
	cudaMemcpy(d_bond_req, bond_req, nbond_types * sizeof(float),
		cudaMemcpyHostToDevice);
	cudaMemcpy(d_bond_k, bond_k, nbond_types * sizeof(float),
		cudaMemcpyHostToDevice);

	

	// Stack 2D array into 1D for transfer to device
	int* h_bond_stuff;
	h_bond_stuff = (int*)calloc(nsSend * MAX_BONDS, sizeof(int));
	int i, j;
	for (i = 0; i < nsSend; i++)
		for (j = 0; j < MAX_BONDS; j++)
			h_bond_stuff[i * MAX_BONDS + j] = bonded_to[i][j];

	cudaMemcpy(d_bonded_to, h_bond_stuff, nsSend * MAX_BONDS * sizeof(int),
		cudaMemcpyHostToDevice);

	for (i = 0; i < nsSend; i++)
		for (j = 0; j < MAX_BONDS; j++)
			h_bond_stuff[i * MAX_BONDS + j] = bond_type[i][j];

	cudaMemcpy(d_bond_type, h_bond_stuff, nsSend * MAX_BONDS * sizeof(int),
		cudaMemcpyHostToDevice);

	for (i = 0; i < nsSend; i++)
		for (j = 0; j < MAX_BONDS; j++)
			h_bond_stuff[i * MAX_BONDS + j] = bond_style[bond_type[i][j]];

	cudaMemcpy(d_bond_style, h_bond_stuff, nsSend * MAX_BONDS * sizeof(int),
		cudaMemcpyHostToDevice);

	free(h_bond_stuff);


    
    
    // Send angle info to device if needed
    if ( n_total_angles > 0 ) {
        cudaMemcpy(d_angle_k, angle_k, nangle_types * sizeof(float), 
            cudaMemcpyHostToDevice);
        cudaMemcpy(d_angle_theta_eq, angle_theta_eq, nangle_types * sizeof(float), 
            cudaMemcpyHostToDevice);
        cudaMemcpy(d_angleIntStyle, angleIntStyle, nangle_types * sizeof(int), 
            cudaMemcpyHostToDevice);

        cudaMemcpy(d_n_angles, n_angles, nsSend * sizeof(int),
            cudaMemcpyHostToDevice);
            

        int* h_ang;
        h_ang = (int*)calloc(nsSend * MAX_ANGLES, sizeof(int));

        for ( int i=0 ; i<nsSend ; i++ )
            for ( int j=0 ; j<MAX_ANGLES; j++ )
                h_ang[i * MAX_ANGLES + j] = angle_type[i][j];
        cudaMemcpy(d_angle_type, h_ang, nsSend*MAX_ANGLES * sizeof(int),
            cudaMemcpyHostToDevice);
       
        for ( int i=0 ; i<nsSend ; i++ )
            for ( int j=0 ; j<MAX_ANGLES; j++ )
                h_ang[i * MAX_ANGLES + j] = angle_first[i][j];
        cudaMemcpy(d_angle_first, h_ang, nsSend*MAX_ANGLES * sizeof(int),
            cudaMemcpyHostToDevice);

        for ( int i=0 ; i<nsSend ; i++ )
            for ( int j=0 ; j<MAX_ANGLES; j++ )
                h_ang[i * MAX_ANGLES + j] = angle_mid[i][j];
        cudaMemcpy(d_angle_mid, h_ang, nsSend * MAX_ANGLES * sizeof(int),
            cudaMemcpyHostToDevice);

        for ( int i=0 ; i<nsSend ; i++ )
            for ( int j=0 ; j<MAX_ANGLES; j++ )
                h_ang[i * MAX_ANGLES + j] = angle_end[i][j];
        cudaMemcpy(d_angle_end, h_ang, nsSend*MAX_ANGLES * sizeof(int),
            cudaMemcpyHostToDevice);     
             
    }// if (n_total_angles > 0)



	// Copy grid information
	cudaMemcpy(d_dx, dx, 3 * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Nx, Nx, 3 * sizeof(int), cudaMemcpyHostToDevice);

	// Copy masses, diffusivities
	cudaMemcpy(d_mass, mass, ntypes * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Diff, Diff, ntypes * sizeof(float), cudaMemcpyHostToDevice);

	// Copy charges
	if (Charges::do_charges == 1) 
		cudaMemcpy(d_charges, charges, ns * sizeof(float), cudaMemcpyHostToDevice);
}
