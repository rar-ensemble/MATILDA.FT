// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "tensor_potential_MaierSaupe.h"
#include <iostream>
#include <sstream>
#include "device_utils.cuh"
#include "timing.h"

using namespace std;

__global__ void d_zero_particle_forces(float*, int, int);


void MaierSaupe::CalcForces() {
    MaierSaupe_t_in = int(time(0));

    this->CalcSTensors();

    // First contribution to the force from grad u(rij)
    for ( int j=0 ; j<Dim ; j++ ) {
        // Loop over S-tensor components to convolve with grad_j u(r)
        // Uses the symmetry to halve the calculations
        for ( int k=0 ; k<Dim ; k++ ) {
            for ( int m=k ; m<Dim ; m++ ) {

                // Grab component k, m
                // d_cpx1 = S_km
                d_extractTensorComponent<<<M_Grid, M_Block>>>(d_cpx1, 
                    this->d_S_field, k, m, M, Dim);

                // FFT component k, m
                // d_cpx2 = FFT(S_km)
                cufftExecC2C(fftplan, d_cpx1, d_cpx2, CUFFT_FORWARD);
                check_cudaError("FFTing in Maier-Saupe forces");

                // Multiply component j of d_f_k with d_cpx2
                // Result stored in d_cpx1
                d_prepareForceKSpace<<<M_Grid, M_Block>>>(this->d_f_k, d_cpx2, d_cpx1, j, Dim, M);

                // d_cpx2 = IFFT of above convolution
                cufftExecC2C(fftplan, d_cpx1, d_cpx2, CUFFT_INVERSE);
                check_cudaError("Inverse FFT in Maier-Saupe forces");

                // Store the result in the temporary tensor field
                d_storeTensorComponent<<<M_Grid, M_Block>>>(this->d_tmp_tensor,
                    d_cpx2, k, m, M, Dim);

                // Use the symmetry to store the m,k component too
                if ( k != m ) {
                    d_storeTensorComponent<<<M_Grid, M_Block>>>(this->d_tmp_tensor,
                        d_cpx2, m, k, M, Dim);
                }
            }// m=k:Dim
        }// k=0:Dim

        d_accumulateMSForce1<<<ns_Grid, ns_Block>>>(::d_f, this->d_MS_pair, this->d_tmp_tensor, this->d_ms_S, 
            d_grid_W, d_grid_inds, gvol, j, grid_per_partic, ns, Dim);

    }// j=0:Dim loop over the dimensions of grad u



    // Second contribution from du/dri

    // First, convole S field with u(r)
    for ( int k=0 ; k<Dim ; k++ ) {
        for ( int m=k; m<Dim ; m++ ) {
            // Grab component k, m
            // d_cpx1 = S_km
            d_extractTensorComponent<<<M_Grid, M_Block>>>(d_cpx1, 
                this->d_S_field, k, m, M, Dim);

            // FFT component k, m
            // d_cpx2 = FFT(S_km)
            cufftExecC2C(fftplan, d_cpx1, d_cpx2, CUFFT_FORWARD);
            check_cudaError("FFTing in Maier-Saupe forces");

            // Multiply component j of d_f_k with d_cpx2
            // Result stored in d_cpx1
            d_multiplyComplex<<<M_Grid, M_Block>>>(this->d_u_k, d_cpx2, d_cpx1, M);

            // d_cpx2 = IFFT of above convolution
            cufftExecC2C(fftplan, d_cpx1, d_cpx2, CUFFT_INVERSE);
            check_cudaError("Inverse FFT in Maier-Saupe forces");

            // Store the result in the temporary tensor field
            d_storeTensorComponent<<<M_Grid, M_Block>>>(this->d_tmp_tensor,
                d_cpx2, k, m, M, Dim);

            // Use the symmetry to store the m,k component too
            if ( k != m ) {
                d_storeTensorComponent<<<M_Grid, M_Block>>>(this->d_tmp_tensor,
                    d_cpx2, m, k, M, Dim);
            }            
        }// m=k:Dim
    }// k=0:Dim

    d_accumulateMSForce2<<<ns_Grid, ns_Block>>>(::d_f, d_x, this->d_MS_pair, this->d_tmp_tensor, this->d_ms_u, 
            d_grid_W, d_grid_inds, gvol, grid_per_partic, ns, d_L, d_Lh, Dim);

    MaierSaupe_t_out = int(time(0));
    MaierSaupe_tot_time += MaierSaupe_t_out - MaierSaupe_t_in;
}


// Routine to calculate MS potential energy
// Assumes that forces have been called, meaning:
// this->d_tmp_tensor contains (S*u)(r)
// this->d_S_field is already populated with S(r)
float MaierSaupe::CalcEnergy() {

    d_doubleDotTensorFields<<<M_Grid, M_Block>>>(d_tmp, this->d_tmp_tensor, this->d_S_field, M, Dim);

    cudaMemcpy(tmp, d_tmp, M*sizeof(float), cudaMemcpyDeviceToHost);

    this->energy = -integ_trapPBC(tmp);
    return energy;
}


// Calculate the S Tensors for all of the particles
void MaierSaupe::CalcSTensors() {

    // Calculate particle-level S tensors
    d_calcParticleSTensors<<<ns_Grid, ns_Block>>>(this->d_ms_u, this->d_ms_S, d_x, 
        this->d_MS_pair, d_L, d_Lh, Dim, ns);
    check_cudaError("Calculate particle-level S tensors");

    // Zero the Dim*Dim*M S tensor field
    int biggerM = M*Dim*Dim;
    int bM_Grid = (int)ceil((float(biggerM) / M_Block));
    d_zero_float_vector<<<bM_Grid, M_Block>>>(this->d_S_field, biggerM);
    check_cudaError("Zeroing S field");

    // Map the particle S to the field S
    d_mapFieldSTensors<<<ns_Grid, ns_Block>>>(this->d_S_field, this->d_MS_pair, this->d_ms_S, 
        d_grid_W, d_grid_inds, ns, grid_per_partic, Dim);

    check_cudaError("MapSTensors in Maier-Saupe forces");
}


MaierSaupe::MaierSaupe(istringstream &iss) : Potential(iss) {
	potential_type = "MaierSaupe";
	type_specific_id = num++;

	readRequiredParameter(iss, filename);
	readRequiredParameter(iss, initial_prefactor);
	readRequiredParameter(iss, sigma_squared);

	final_prefactor = initial_prefactor;
	sigma_squared *= sigma_squared;

	ramp_check_input(iss);

}

void MaierSaupe::Initialize() {
    Initialize_Potential();
    Initialize_TensorPotential();
    Allocate();
}


void MaierSaupe::Allocate() {

    // Allocate memory for this potential
    int ns_alloc = ns + extra_ns_memory;
    
    this->MS_pair = (int*) calloc(ns_alloc, sizeof(int));
    cudaMalloc(&this->d_MS_pair, ns_alloc * sizeof(int));
    mem_use += ns_alloc * sizeof(int);
    device_mem_use += ns_alloc * sizeof(int);

    h_Dim_Dim_tensor = (float*) calloc(Dim*Dim, sizeof(float));
    cudaMalloc(&this->d_Dim_Dim_tensor, Dim * Dim* sizeof(float));
    cout << "Allocating " << Dim*Dim << " bytes for tmp_tensor" << endl;

    int size = Dim * ns;
    this->ms_u = (float*) calloc(size, sizeof(float));
    cudaMalloc(&this->d_ms_u, size * sizeof(float));
    mem_use += size * sizeof(float);
    device_mem_use += size * sizeof(float);

    size = Dim * Dim * ns;
    this->ms_S = (float*) calloc(size, sizeof(float));
    cudaMalloc(&this->d_ms_S, size * sizeof(float));
    mem_use += size * sizeof(float);
    device_mem_use += size * sizeof(float);

    size = Dim * Dim * M;
    this->S_field = ( float* ) calloc(size, sizeof(float));
    cudaMalloc(&this->d_S_field, size * sizeof(float));
    cudaMalloc(&this->d_tmp_tensor, size * sizeof(float));
    mem_use += size * sizeof(float);
    device_mem_use += 2 * size * sizeof(float);

    this->allocated = true;
    // End memory allocation
    check_cudaError("Allocating memory for Maier Saupe");


    // Set all partners initially to -1
    for ( int i=0 ; i<ns ; i++ ) this->MS_pair[i] = -1;



    this->read_lc_file(this->filename);


    // Initialize the Gaussian potential
    // Initialize in k-space

    init_device_gaussian<<<M_Grid, M_Block>>>(this->d_u_k, this->d_f_k,
        initial_prefactor, this->sigma_squared, d_L, M, d_Nx, Dim);

    init_device_gaussian<<<M_Grid, M_Block>>>(this->d_master_u_k, this->d_master_f_k,
        1, this->sigma_squared, d_L, M, d_Nx, Dim);

    // Inverse transform into real-space
    cufftExecC2C(fftplan, this->d_u_k, d_cpx1, CUFFT_INVERSE);

    // Store real-space version
    d_complex2real<<<M_Grid, M_Block>>>(d_cpx1, this->d_u, M);

    // Store the potential on the host, too
    cudaMemcpy(this->u, this->d_u, M*sizeof(float), cudaMemcpyDeviceToHost);


}


void MaierSaupe::read_lc_file(string name) {
    FILE *inp;
    inp = fopen(name.c_str(), "r");
    if ( inp == NULL ) 
        die("MaierSaupe input file not found!");

    int id1, id2, di;

    // Reads the number of MaierSaupe pairs
    (void)!fscanf(inp, "%d\n", &nms);

    // Scans the rest of the file for all the MS pairs
    // Note the file is expected to be 1-indexed, so 
    // the -1 below is to shift to 0 indexing.
    for ( int i=0 ; i<nms ; i++ ) {
        (void)!fscanf(inp, "%d %d %d\n", &di, &id1, &id2);

        this->MS_pair[id1-1] = id2-1;
    }

    fclose(inp);

    // Copy the ms list to the device
    cudaMemcpy(this->d_MS_pair, this->MS_pair, ns*sizeof(int), cudaMemcpyHostToDevice);
}


MaierSaupe::MaierSaupe() : TensorPotential() {
    type1 = -1; 
    type2 = -1; 
}

MaierSaupe::~MaierSaupe(){}

void MaierSaupe::ramp_check_input(istringstream& iss){

    if (iss.fail()){
        die("Error during input script; failed to properly read:\n" + iss.str());
    }

    string convert;
    iss >> convert;

    if (!iss.fail()){
        if (convert == "ramp") {
            ramp = true;
            iss >> final_prefactor;
            if(iss.fail()) die("no final prefactor specified");
            cout << "Ramping prefactor of " <<potential_type<< " style from " << initial_prefactor \
                << " to " << final_prefactor << endl;

            cout << "Estimated per time step change: " << \
                (final_prefactor - initial_prefactor) / (prod_steps)
                << endl;

        }
        else 
            die("Invalid keyword: " + convert);
    }


}

float MaierSaupe::CalculateOrderParameter(){

    CalcSTensors();
    check_cudaError("Calculate S tensor in CalculateOrderParameter");

    // Average the particle S tensors to the device

    // Zero the Dim*Dim*M S tensor field
    int DD = Dim*Dim;

    d_zero_float_vector<<<1, DD>>>(d_Dim_Dim_tensor, Dim*Dim);
    check_cudaError("Zero d_tmp_tensor in CalculateOrderParameter");

    d_SumAndAverageSTensors<<<ns_Grid, ns_Block>>>(this->d_ms_S, this->d_Dim_Dim_tensor, this->d_MS_pair, Dim, ns);
    check_cudaError("Average STensors in CalculateOrderParameter");

    // Copy Dim*Dim float values from d_tmp_tensor to h_tmp_tensor

    cudaMemcpy(this->h_Dim_Dim_tensor, this->d_Dim_Dim_tensor, Dim*Dim*sizeof(float), cudaMemcpyDeviceToHost);

    check_cudaError("Copy d_tmp_tensor to host in CalculateOrderParameter");


    return CalculateMaxEigenValue(&h_Dim_Dim_tensor[0]) / float(nms);
}

float MaierSaupe::CalculateOrderParameterGridPoints(){

    // Allocate a static vector for floats of length M

    static std::vector<float> h_grid_W(M, 0);

    std::fill(h_grid_W.begin(), h_grid_W.end(), 0);

    CalcSTensors();
    check_cudaError("Calculate S tensor in CalculateOrderParameterGridPoints");

    // Zero the Dim*Dim*M S tensor field
    int DDM = Dim*Dim*M;

    cudaMemcpy(this->h_S_field, this->d_S_field, DDM*sizeof(float), cudaMemcpyDeviceToHost);

    check_cudaError("Copy d_D_field to host in CalculateOrderParameterGridPoints");

    cudaMemcpy(h_grid_W.data(), d_grid_W, M*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i=0; i<M; i++){
        CalculateMaxEigenValue(&h_S_field[i * Dim*Dim]) / float(h_grid_W[i]);
    }
}



void MaierSaupe::ReportEnergies(int& die_flag){
    static int counter = 0;
	static string reported_energy = "";
	static string reported_order = "";

    string tmp_energy =  " " + to_string(energy);
    string tmp_order =  " " + to_string(CalculateOrderParameter());

    dout << tmp_energy;
    dout << tmp_order;

	reported_energy += tmp_energy;
	reported_order += tmp_order;
	if (std::isnan(energy)) die_flag = 1 ;
    

    if (++counter == num){
        cout << " UMaierSaupe:" + reported_energy;
        cout << " LambdaMaierSaupe:" + reported_order;
        counter=0;
		reported_energy.clear();
		reported_order.clear();
    }

}

int MaierSaupe::num = 0;
