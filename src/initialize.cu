// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "timing.h"
#include <string>
#include <sstream>
#include <cmath>
#include <algorithm>
#include "random.h"

using namespace std;

void allocate_device_memory(void);
void allocate_device_potentials(void);
void send_box_params_to_device(void);
void send_3n_to_device(float**, float*);
void allocate_grid_memory(void);
void read_input(void);
void cuda_collect_x(void);
void init_binary_output(void);
__global__ void init_dev_rng(unsigned int, curandState*, int);
__global__ void d_assignValueR(float*, float, int, int);



void initialize() {
	step = 0;
	mem_use = 0;
    extra_ns_memory = 0;
	
	// Global unit complex value
	I = complex<float>(0.0f, 1.0f);
	
	
	read_input();
	cout << "Input file read!" << endl;

	Potential::determine_types_to_fft();

	M = 1;
	grid_per_partic = 1;
	gvol = 1.f;
	for (int j = 0; j < Dim; j++) {
		M *= Nx[j];
		dx[j] = L[j] / float(Nx[j]);
		gvol *= dx[j];
		grid_per_partic *= (pmeorder + 1);
	}
	
	// Define the number of pressure tensor components
	// if Dim == 2: 0=xx, 1=yy, 2=xy
	// If Dim == 3: 0=xx, 1=yy, 2=zz, 3=xy, 4=xz, 5=yz
	n_P_comps = int(Dim*(Dim +1)) / 2;

	print_tot_time = 0;
	bond_tot_time = 0;
	compute_tot_time = 0;
    MaierSaupe_tot_time = 0;
    extraForce_tot_time = 0;
	nList_tot_time = 0;
    DPD_time = 0;
    nl_time = 0;

	device_mem_use = 0;
	allocate_grid_memory();
	cout << "Grid memory allocated! " << endl;

	allocate_device_memory();



	// Sends box geometry, bond topology, other
	// one-time communication information
	send_box_params_to_device();
	cout << "Box parameters sent to device" << endl;


	allocate_device_potentials();
	cout << "Device potentials initialized" << endl;

	// Send initial box coordinates
	send_3n_to_device(::x, d_x);
	cout << "Initial box coordinates sent!" << endl;
	

	cout << "Device memory allocated: " << float(device_mem_use) / powf(10.0f, 6)
		<< " MB\n";

	cuda_collect_x();

	//if (bin_freq != 0)
	//	init_binary_output();

	// Sort Pair Styles so things print out nicely later
	std::sort(Potentials.begin(), Potentials.end(), [](Potential* a, Potential* b) {
		if (a->potential_type!= b->potential_type) return a->potential_type < b->potential_type;
		return a->type_specific_id < b->type_specific_id;
	});
	
}





int ns_alloc;

void allocate_device_memory() {

	device_mem_use = 0;

	ns_Block = threads;
	ns_Grid = (int)ceil((float)(ns) / ns_Block);
	printf("ns: %d, ns_Grid: %d\n", ns, ns_Grid);


	M_Block = threads;
	M_Grid = (int)ceil((float)(M) / M_Block);
	cout << "M: " << M << ", M_Grid: " << M_Grid << endl;

    idum = RAND_SEED;

    ns_alloc = ns + extra_ns_memory;
	int size = ns_alloc * Dim * sizeof(float);
	cudaMalloc(&d_x, size);
	cudaMalloc(&d_f, size);
	cudaMalloc(&d_xo, size);
	cudaMalloc(&d_v, size);
	cudaMalloc(&d_3n_tmp, size);

    cudaMalloc(&d_molecID, ns_alloc * sizeof(int));

	device_mem_use += size * 5;

	if (Integrator::using_GJF) {
		cudaMalloc(&d_xo, size);
		cudaMalloc(&d_prev_noise, size);

		d_assignValueR<<<ns_Grid, ns_Block>>>(d_prev_noise, 0.0f, Dim, ns);

		check_cudaError("allocating memory");

		device_mem_use += 2 * size;
	}



	// Initialize random number seeds on each thread
	cudaMalloc(&d_states, ns_alloc * Dim * sizeof(curandState));
	init_dev_rng<<<ns_Grid, ns_Block>>>(RAND_SEED, d_states, ns_alloc);

	device_mem_use += ns_alloc * Dim * sizeof(curandState);

	cudaMalloc(&d_mass, ntypes * sizeof(float));
	cudaMalloc(&d_Diff, ntypes * sizeof(float));

	// Allocate box parameters on device
	cudaMalloc(&d_L, 6 * sizeof(float));
	cudaMalloc(&d_Lh, 6 * sizeof(float));

	device_mem_use += sizeof(float) * (3 + 3);

	cudaMalloc(&d_typ, ns_alloc * sizeof(int));
	device_mem_use += sizeof(int) * ns_alloc;

	// Bond parameters on device
	cudaMalloc(&d_n_bonds, ns_alloc * sizeof(int));
	cudaMalloc(&d_bonded_to, ns_alloc * MAX_BONDS * sizeof(int));
	cudaMalloc(&d_bond_type, ns_alloc * MAX_BONDS * sizeof(int));
	cudaMalloc(&d_bond_style, ns_alloc * MAX_BONDS * sizeof(int));
	device_mem_use += sizeof(int) * (ns_alloc + ns_alloc * MAX_BONDS * 2 );

	cudaMalloc(&d_bond_req, nbond_types * sizeof(float));
	cudaMalloc(&d_bond_k, nbond_types * sizeof(float));

    device_mem_use += sizeof(float) * nbond_types * 2;
	
	cudaMalloc(&d_bondE, ns_alloc * sizeof(float));
	cudaMalloc(&d_bondVir, ns_alloc * n_P_comps * sizeof(float));

	device_mem_use += ns_alloc * (n_P_comps + 1) * sizeof(float);

	if (n_total_angles > 0) {
        cudaMalloc(&d_angle_k, nangle_types * sizeof(float));
        cudaMalloc(&d_angle_theta_eq, nangle_types * sizeof(float));
        cudaMalloc(&d_angleIntStyle, nangle_types * sizeof(int));
        device_mem_use += sizeof(float) * nangle_types * 2;

        cudaMalloc(&d_n_angles, ns_alloc * sizeof(int));
        cudaMalloc(&d_angle_first, ns_alloc * MAX_ANGLES * sizeof(int));
        cudaMalloc(&d_angle_mid, ns_alloc * MAX_ANGLES * sizeof(int));
        cudaMalloc(&d_angle_end, ns_alloc * MAX_ANGLES * sizeof(int));
        cudaMalloc(&d_angle_type, ns_alloc * MAX_ANGLES * sizeof(int));
        device_mem_use += ns_alloc * ( 1 + 4 * MAX_ANGLES ) * sizeof(int);
		//die("Angles not yet set up on device!");
	}

	cudaMalloc(&d_Nx, 3 * sizeof(int));
	device_mem_use += sizeof(int) * 3;

	cudaMalloc(&d_dx, 3 * sizeof(float));
	cudaMalloc(&d_tmp, M * sizeof(float));
	cudaMalloc(&d_tmp2, M * sizeof(float));
	cudaMalloc(&d_all_rho, ntypes * M * sizeof(float));


	GRID_STATE = (int*) calloc(ntypes,sizeof(int));
	
	std::cout << "Grid state " << std::endl;
	for (int i = 0; i < ntypes; ++i){
		GRID_STATE[i] = 0;
	}

	d_calculated_rho_all = (cufftComplex**)calloc(ntypes, sizeof(cufftComplex*));
	for (int i = 0; i < ntypes; i++) {
		 cudaMalloc(&d_calculated_rho_all[i], M * sizeof(cufftComplex));
	}

	device_mem_use += sizeof(cufftComplex) * (ntypes * M);


	cudaMalloc(&d_all_fx, ntypes * M * sizeof(float));
	cudaMalloc(&d_all_fy, ntypes * M * sizeof(float));
	device_mem_use += sizeof(float) * (2 * M + ntypes * M * 3 + 3);

	if (Dim == 3) {
		cudaMalloc(&d_all_fz, ntypes * M * sizeof(float));
		device_mem_use += sizeof(float*) * (ntypes * M);
	}


	cudaMalloc(&d_cpx1, M * sizeof(cufftComplex));
	cudaMalloc(&d_cpx2, M * sizeof(cufftComplex));
	cudaMalloc(&d_cpxx, M * sizeof(cufftComplex));
	cudaMalloc(&d_cpxy, M * sizeof(cufftComplex));
	cudaMalloc(&d_cpxz, M * sizeof(cufftComplex));
	device_mem_use += sizeof(cufftComplex) * (2 * M);

	cudaMalloc(&d_nan, sizeof(bool));
	device_mem_use += sizeof(bool);

	cudaMalloc(&d_charges, ns_alloc * sizeof(float));

	cudaMalloc(&d_grid_W, ns_alloc * grid_per_partic * sizeof(float));
	cudaMalloc(&d_grid_inds, ns_alloc * grid_per_partic * sizeof(int));
	device_mem_use += ns_alloc * grid_per_partic * (sizeof(float) + sizeof(int));

	if (Dim == 2)
		cufftPlan2d(&fftplan, Nx[1], Nx[0], CUFFT_C2C);
	else if (Dim == 3)
		cufftPlan3d(&fftplan, Nx[2], Nx[1], Nx[0], CUFFT_C2C);

	cudaStreamCreateWithFlags(&stream1,cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&stream2,cudaStreamNonBlocking);
	cudaStreamCreateWithFlags(&stream3,cudaStreamNonBlocking);

}

void allocate_device_potentials(void) {

	for (auto Iter: Potentials)
		Iter->Initialize();
	}

void allocate_grid_memory(void) {
	tmp = (float*)calloc(M, sizeof(float*));
	tmp2 = (float*)calloc(M, sizeof(float*));
	cpx1 = (cufftComplex*)calloc(M, sizeof(cufftComplex));
	cpx2 = (cufftComplex*)calloc(M, sizeof(cufftComplex));
	k_tmp = (complex<float>*) calloc(M, sizeof(complex<float>));


	all_rho = (float*)calloc(M * ntypes, sizeof(float));
	
	Components = new FieldComponent[ntypes];

  for (auto Iter : Computes) {
    Iter->allocStorage();
  }

}

void allocate_host_particles() {

    int ns_alloc = ns + extra_ns_memory;

    if ( extra_ns_memory ) {
        cout << "Allocating memory for " << extra_ns_memory << " extra sites!" << endl;
    }

	n_P_comps = int (Dim*(Dim+1))/2;

	x = (float**)calloc(ns_alloc, sizeof(float*));
	xo = (float**)calloc(ns_alloc, sizeof(float*));
	v = (float**)calloc(ns_alloc, sizeof(float*));
	f = (float**)calloc(ns_alloc, sizeof(float*));
	if (Charges::do_charges == 1) charges = (float*)calloc(ns_alloc, sizeof(float));

	for (int i = 0; i < ns_alloc; i++) {
		x[i] = (float*)calloc(Dim, sizeof(float));
		xo[i] = (float*)calloc(Dim, sizeof(float));
		v[i] = (float*)calloc(Dim, sizeof(float));
		f[i] = (float*)calloc(Dim, sizeof(float));
	}

	h_ns_float = (float*)calloc(ns_alloc * Dim, sizeof(float));

	partic_bondE = (float*)calloc(ns_alloc, sizeof(float));
	partic_bondVir = (float*)calloc(ns_alloc * n_P_comps, sizeof(float));
	bondVir = (float*)calloc(n_P_comps, sizeof(float));
	angleVir = (float*)calloc(n_P_comps, sizeof(float));

	tp = (int*)calloc(ns_alloc, sizeof(int));
	molecID = (int*)calloc(ns_alloc, sizeof(int));

	mass = (float*)calloc(ntypes, sizeof(float));
	Diff = (float*)calloc(ntypes, sizeof(float));

	// Set default diffusivities
	for (int i = 0; i < ntypes; i++)
		Diff[i] = 1.0;

	Ptens = (float*)calloc(n_P_comps, sizeof(float));

	// NOTE: Assumes that a particle is bonded to a maximum 
	// of MAX_BONDS particles
	n_bonds = (int*)calloc(ns_alloc, sizeof(int));
	n_angles = (int*)calloc(ns_alloc, sizeof(int));
	bonded_to = (int**)calloc(ns_alloc, sizeof(int*));
	bond_type = (int**)calloc(ns_alloc, sizeof(int*));
	angle_first = (int**)calloc(ns_alloc, sizeof(int*));
	angle_mid = (int**)calloc(ns_alloc, sizeof(int*));
	angle_end = (int**)calloc(ns_alloc, sizeof(int*));
	angle_type = (int**)calloc(ns_alloc, sizeof(int*));

	for (int i = 0; i < ns_alloc; i++) {
		bonded_to[i] = (int*)calloc(MAX_BONDS, sizeof(int));
		bond_type[i] = (int*)calloc(MAX_BONDS, sizeof(int));

		angle_first[i] = (int*)calloc(MAX_ANGLES, sizeof(int));
		angle_mid[i] = (int*)calloc(MAX_ANGLES, sizeof(int));
		angle_end[i] = (int*)calloc(MAX_ANGLES, sizeof(int));
		angle_type[i] = (int*)calloc(MAX_ANGLES, sizeof(int));
	}

	mem_use += ns_alloc * (2 * MAX_BONDS + 4 * MAX_ANGLES + 2) * sizeof(int);

	bond_k = (float*)calloc(nbond_types, sizeof(float));
	bond_req = (float*)calloc(nbond_types, sizeof(float));
	bond_style = (int*)calloc(nbond_types, sizeof(int));

	angleStyle.resize(nangle_types);
	angleIntStyle = (int*) calloc(nangle_types, sizeof(int));
	angle_k = (float*)calloc(nangle_types, sizeof(float));
	angle_theta_eq = (float*)calloc(nangle_types, sizeof(float));

	if (Charges::do_charges == 1) {
		electrostatic_energy = (float*)calloc(1, sizeof(float));
		electrostatic_energy_direct_computation = (float*)calloc(1, sizeof(float));
		//charges_over_M = (float*)calloc(ns, sizeof(float));
	}

}


