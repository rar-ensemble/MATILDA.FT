// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "potential_gaussian.h"
#include "device_utils.cuh"
#include <iostream>
#include <fstream>
using namespace std;

__global__ void init_device_gaussian(cufftComplex*, cufftComplex*, 
    float, float, const float*,
    const int, const int*, const int);
__global__ void d_multiply_cufftCpx_scalar(cufftComplex*, float, int);
__global__ void d_complex2real(cufftComplex*, float*, int);
__global__ void d_extractForceComp(cufftComplex*, cufftComplex*,
    const int, const int, const int);
__global__ void d_insertForceCompC2R(float*, cufftComplex*, const int,
    const int, const int);

void Gaussian::Initialize() {
    Initialize_Potential();

    printf("Setting up Gaussian pair style..."); fflush(stdout);
 
    init_device_gaussian<<<M_Grid, M_Block>>>(this->d_u_k, this->d_f_k,
        initial_prefactor, sigma_squared, d_L, M, d_Nx, Dim);

    init_device_gaussian<<<M_Grid, M_Block>>>(
        this->d_master_u_k, this->d_master_f_k,
        1.0f, sigma_squared, d_L, M, d_Nx, Dim);

    cufftExecC2C(fftplan, this->d_u_k, d_cpx1, CUFFT_INVERSE);
    d_complex2real<<<M_Grid, M_Block>>>(d_cpx1, this->d_u, M);

    for (int j = 0; j < Dim; j++) {
        d_extractForceComp<<<M_Grid, M_Block>>>(d_cpx1, this->d_f_k, j, Dim, M);
        cufftExecC2C(fftplan, d_cpx1, d_cpx1, CUFFT_INVERSE);
        d_insertForceCompC2R<<<M_Grid, M_Block>>>(this->d_f, d_cpx1, j, Dim, M);
    }

    float k2, kv[3];

    // Define the potential and the force in k-space
    for (int i = 0; i < M; i++) {
        k2 = get_k(i, kv, Dim);

        this->u_k[i] = initial_prefactor * exp(-k2 * sigma_squared / 2.0f);
        

        for (int j = 0; j < Dim; j++) {
            this->f_k[j][i] = -I * kv[j] * this->u_k[i];
        }
            
    }
    
  
    
    InitializeVirial();
    
    printf("done!\n"); fflush(stdout);
}


Gaussian::Gaussian() {

}

Gaussian::Gaussian(istringstream &iss) : Potential(iss) {
	potential_type = "Gaussian";
	type_specific_id = num++;
    SAME_TYPE = 0;

	readRequiredParameter(iss, type1);
	readRequiredParameter(iss, type2);
	readRequiredParameter(iss, initial_prefactor);
	readRequiredParameter(iss, sigma_squared);
    if (type1 == type2){
        SAME_TYPE = 1;
    }

	// iss >> type1 >> type2 >> initial_prefactor >> sigma_squared;
    // if (iss.fail()) std::cout << "here" << std::endl;

	type1 -= 1;
	type2 -= 1;

	final_prefactor = initial_prefactor;

	sigma_squared *= sigma_squared;

	check_types();

	ramp_check_input(iss);

}

Gaussian::~Gaussian() {

}


__global__ void init_device_gaussian(cufftComplex* uk, cufftComplex* fk,
    float Ao, float sigma2,
    const float* dL, const int M, const int* Nx, const int Dim) {

    const int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= M)
        return;

    float k2, kv[3];

    k2 = d_get_k(ind, kv, dL, Nx, Dim);
    uk[ind].x = Ao * exp(-k2 * sigma2 / 2.0f);
    uk[ind].y = 0.f;

    for (int j = 0; j < Dim; j++) {
        fk[ind * Dim + j].x = 0.f;
        fk[ind * Dim + j].y = -kv[j] * uk[ind].x;
    }
}

void Gaussian::ReportEnergies(int& die_flag){
    static int counter = 0;
    static string reported_energy = "";
    reported_energy += " " + to_string(energy) ;
	if (std::isnan(energy)) die_flag = 1 ;
    
    if (++counter == num){
        dout << reported_energy;
        cout << " Ugauss: " + reported_energy;
        counter=0;
		reported_energy.clear();
    }
}

int Gaussian::num = 0;
