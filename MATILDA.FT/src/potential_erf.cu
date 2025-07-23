// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "potential_erf.h"
#include "device_utils.cuh"

using namespace std;

__global__ void init_device_erf(float*,	float, float, float, 
	const float*, const float*, const float*, 
	const int, const int*, const int);
__global__ void init_device_erf_kspace(cufftComplex*, cufftComplex*, float, float, float, 
	const float*, const int, const int*, const int);
__global__ void d_real2complex(float*, cufftComplex*, int);
__global__ void d_kSpaceMakeForce(cufftComplex*, cufftComplex*,
	const float*, const int*, const int, const int);
__global__ void d_extractForceComp(cufftComplex*, cufftComplex*, int, int, int);
__global__ void d_insertForceCompC2R(float*, cufftComplex*, const int,
	const int, const int);
__global__ void d_divideByDimension(cufftComplex*, int);

__global__ void d_complex2real(cufftComplex*, float*, int);

void Erf::Initialize() {
	cout << "Setting up ERF pairstyle...";
	fflush(stdout);

    Initialize_Potential();

	init_device_erf_kspace<<<M_Grid,M_Block>>>(this->d_u_k, this->d_f_k, initial_prefactor, sigma_squared, Rp,
		d_L,M, d_Nx, Dim);

	init_device_erf_kspace<<<M_Grid,M_Block>>>(
        this->d_master_u_k, this->d_master_f_k,
        1.0f, sigma_squared, Rp, d_L, M, d_Nx, Dim);

	cufftExecC2C(fftplan, this->d_u_k, d_cpx1, CUFFT_INVERSE);
	d_complex2real << <M_Grid, M_Block >> > (d_cpx1, this->d_u, M);

	for (int j = 0; j < Dim; j++) {
		d_extractForceComp << <M_Grid, M_Block >> > (d_cpx1, this->d_f_k, j, Dim, M);
		cufftExecC2C(fftplan, d_cpx1, d_cpx1, CUFFT_INVERSE);
		d_insertForceCompC2R << <M_Grid, M_Block >> > (this->d_f, d_cpx1, j, Dim, M);
	}

	// Define the potential and the force in k-space

	float k2, kv[3], k;
	float temp;
	

	for (int i = 0; i < M; i++) {
		k2 = get_k(i, kv, Dim);
		k = sqrt(k2);

		if (k2 == 0) {
			this->u_k[i] = initial_prefactor * // prefactor
				// exp(-k2 * sigma_squared * 0.5f) * //Gaussian contribution = 1
				PI4 * Rp * Rp * Rp / 3 *   // step function contribution for 1
				PI4 * Rp * Rp * Rp / 3;   // step function contribution for 2
		}
		else
		{
			//FFT of step function 
			temp = PI4 * (sin(Rp * k) - Rp * k * cos(Rp * k)) / (k2 * k);

			this->u_k[i] = initial_prefactor * // prefactor
				exp(-k2 * sigma_squared * 0.5f) * //Gaussian contribution of both
				temp * // step function for 1
				temp; // step function for the other
		}

		for (int j = 0; j < Dim; j++) {
			this->f_k[j][i] = -I * kv[j] * this->u_k[i];
		}

	}

    InitializeVirial();
	cout << "Done!" << endl;
}


Erf::Erf(istringstream &iss) : Potential{iss} {
	potential_type = "Erf";
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

Erf::Erf() {

}

Erf::~Erf() {

}


/* This code defines the erf function on the device,
which will then be Fourier transformed to get the force.
*/
__global__ void init_device_erf(float* u,
	float Ao, float Rp, float xi,
	const float* dL, const float* dLh, const float* ddx,
	const int dM, const int* dNx, const int dDim) {

	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= dM)
		return;

	float ro[3], ri[3], dr[3];
	for (int j = 0; j < dDim; j++)
		ro[j] = 0.f;

	d_get_r(ind, ri, dNx, ddx, dDim);

	float mdr2 = d_pbc_mdr2(ro, ri, dr, dL, dLh, dDim);
	float mdr = sqrtf(mdr2);

	float arg = (mdr - Rp) / xi;
	u[ind] = Ao * (1.0f - erff(arg));
}


/* This code defines the convolved erf function on the device in Fourier space.
 * We have fourier the Gaussian and step function contributions for each erf func.
 * The definition of xi is sqrt(2) sigma based off the comparison of 
 * 1D definition of convolv of box with gauss to get erfc and 
 * the definition we used in our papers
*/
__global__ void init_device_erf_kspace(cufftComplex* uk,
	cufftComplex* fk, float Ao, float sigma2, float Rp,
	const float* dL, const int dM, const int* dNx, const int dDim) {

	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= dM)
		return;

	float k2, kv[3], k;

    k2 = d_get_k(ind, kv, dL, dNx, dDim);
	k = sqrt(k2);
	
	if (k2 == 0) {
		uk[ind].x = Ao *				// prefactor
			//exp(-k2 * sigma_squared * 0.5f )* //Gaussian contribution = 1
			PI4 * Rp * Rp * Rp / 3*   // step function contribution for 1
			PI4 * Rp * Rp * Rp / 3;   // step function contribution for 2
	}
	else
	{
		//FFT of step function 
		float temp = PI4 * (sin(Rp * k) - Rp * k * cos(Rp * k)) / (k2 * k);

		uk[ind].x = Ao *				//prefactor
			exp(-k2 * sigma2 * 0.5f) * //Gaussian contribution of both
			temp * // step function for 1
			temp ; // step function for the other
	}
	uk[ind].y = 0.f;

    for (int j = 0; j < dDim; j++) {
        fk[ind * dDim + j].x = 0.f;
        fk[ind * dDim + j].y = -kv[j] * uk[ind].x;
    }

}

void Erf::ReportEnergies(int& die_flag){
    static int counter = 0;
	static string reported_energy = "";
	reported_energy +=  " " + to_string(energy);
	if (std::isnan(energy)) die_flag = 1 ;
    
    if (++counter == num){
        dout << reported_energy;
        cout << " Uerf:" + reported_energy;
        counter=0;
		reported_energy.clear();
    }
}

int    Erf::num = 0;