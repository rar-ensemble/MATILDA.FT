// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "potential_gaussian_erf.h"
#include "device_utils.cuh"
#include <iostream>
#include <fstream>
using namespace std;

__global__ void init_device_gaussian_erf(cufftComplex*, cufftComplex*, 
    float, float, float, const float*,
    const int, const int*, const int);
__global__ void d_multiply_cufftCpx_scalar(cufftComplex*, float, int);
__global__ void d_complex2real(cufftComplex*, float*, int);
__global__ void d_extractForceComp(cufftComplex*, cufftComplex*,
    const int, const int, const int);
__global__ void d_insertForceCompC2R(float*, cufftComplex*, const int,
    const int, const int);

void GaussianErf::Initialize() {
    Initialize_Potential();

    printf("Setting up Gaussian-Erf pair style..."); fflush(stdout);
 
    init_device_gaussian_erf<<<M_Grid, M_Block>>>(
		this->d_u_k, this->d_f_k,
        initial_prefactor, sigma_squared, Rp, d_L, M, d_Nx, Dim);

    init_device_gaussian_erf<<<M_Grid, M_Block>>>(
        this->d_master_u_k, this->d_master_f_k,
        1, sigma_squared, Rp, d_L, M, d_Nx, Dim);

    cufftExecC2C(fftplan, this->d_u_k, d_cpx1, CUFFT_INVERSE);
    d_complex2real<<<M_Grid, M_Block>>>(d_cpx1, this->d_u, M);

    for (int j = 0; j < Dim; j++) {
        d_extractForceComp<<<M_Grid, M_Block>>>(d_cpx1, this->d_f_k, j, Dim, M);
        cufftExecC2C(fftplan, d_cpx1, d_cpx1, CUFFT_INVERSE);
        d_insertForceCompC2R<<<M_Grid, M_Block>>>(this->d_f, d_cpx1, j, Dim, M);
    }

    float k2, kv[3], k;

    // Define the potential and the force in k-space
    for (int i = 0; i < M; i++) {
        k2 = get_k(i, kv, Dim);
		k = sqrt(k2);

		if (k2 == 0) {
			this->u_k[i] = initial_prefactor * // prefactor
				// exp(-k2 * sigma2 * 0.5f) * //Gaussian contribution  = 1
				PI4 * Rp * Rp * Rp / 3;   // step function contribution
		}
		else
		{
			this->u_k[i] = initial_prefactor * // prefactor
				exp(-k2 * sigma_squared * 0.5f) * //Gaussian contribution of both
				PI4 * (sin(Rp * k) - Rp * k * cos(Rp * k)) / (k2 * k); // step function contribution
		}

        for (int j = 0; j < Dim; j++) {
            this->f_k[j][i] = -I * kv[j] * this->u_k[i];
        }
    }
    
    
    InitializeVirial();
    
    printf("done!\n"); fflush(stdout);
}


GaussianErf::GaussianErf(istringstream &iss) : Potential(iss) {
	potential_type = "GaussianErf";
	type_specific_id = num++;

	readRequiredParameter(iss, type1);
	readRequiredParameter(iss, type2);
	readRequiredParameter(iss, initial_prefactor);
	readRequiredParameter(iss, Rp);
	readRequiredParameter(iss, sigma_squared);

	// iss >> type1 >> type2 >> initial_prefactor >> Rp >> sigma_squared;

	type1 -= 1;
	type2 -= 1;

	final_prefactor = initial_prefactor;

	sigma_squared *= sigma_squared;

	check_types();

	ramp_check_input(iss);
}



GaussianErf::GaussianErf() {

}

GaussianErf::~GaussianErf() {

}


__global__ void init_device_gaussian_erf(cufftComplex* uk, cufftComplex* fk,
    float Ao, float sigma2, float Rp,
    const float* dL, const int M, const int* Nx, const int Dim) {


	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= M)
		return;

	float k2, kv[3], k;

	k2 = d_get_k(ind, kv, dL, Nx, Dim);
	k = sqrt(k2);

	if (k2 == 0) {
		uk[ind].x = Ao *				// prefactor
			//exp(-k2 * sigma2 * 0.5f) * //Gaussian contribution = 1
			PI4 * Rp * Rp * Rp / 3;   // step function contribution of erfc
	}
	else
	{

		uk[ind].x = Ao *				//prefactor
			exp(-k2 * sigma2* 0.5f) * //Gaussian contribution of both
			PI4 * (sin(Rp * k) - Rp * k * cos(Rp * k)) / (k2 * k);
		// step function for erfc only
	}
	uk[ind].y = 0.f;
	for (int j = 0; j < Dim; j++) {
		fk[ind * Dim + j].x = 0.f;
		fk[ind * Dim + j].y = -kv[j] * uk[ind].x;
	}

}

void GaussianErf::ReportEnergies(int &die_flag){
	static int counter = 0;
	static string reported_energy = "";
	reported_energy += " " + to_string(energy) ;
	if (std::isnan(energy)) die_flag = 1 ;
    
    if (++counter == num){
        dout << reported_energy;
        cout << "Ugausserf: " + reported_energy;
        counter=0;
		reported_energy.clear();
    }
}

int    GaussianErf::num = 0;
