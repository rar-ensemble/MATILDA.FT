// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "integrator_GJF.h"

GJF::~GJF(){return;}

GJF::GJF(std::istringstream& iss) : Integrator(iss) {
    
    using_GJF = 1;
}

void GJF::Integrate_2(){

    d_GJF_integrator<<<group->GRID, group->BLOCK>>>(d_x, d_xo, d_f,
        d_prev_noise, d_mass, d_Diff, d_typ,
        delt, noise_mag, d_L, d_Lh, group->d_index.data(),
        group->nsites, Dim, d_states);
}

__global__ void d_GJF_integrator(float* x, float* xo, 
	float* f, float* old_noise,
	float *mass, float *diff, int *typ, float delt, float noise_mag, 
	float* L, float* Lh, 
	thrust::device_ptr<int> d_index,
	int ns, 
	int Dim, 
	curandState* d_states){

	int list_ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (list_ind >= ns)
		return;
	float total_disp, disp_sq;
	int ind = d_index[list_ind];

	curandState l_state;

	l_state = d_states[ind];

	int itype = typ[ind];

	float b = mass[itype] / (mass[itype] + delt / (2.0f * diff[itype]));
	
	float a = (1.0f - delt / (2.0f * diff[itype] * mass[itype])) /
		(1.0f + delt / (2.0f * diff[itype] * mass[itype]));
	
	float delt2 = delt * delt;

	for (int j = 0; j < Dim; j++) {
		int aind = ind * Dim + j;

		// Not sure this should be divided by diff[itype]
		float new_noise = noise_mag / diff[itype] * curand_normal(&l_state);
		
		float xtmp = x[aind];

		if (xo[aind] - x[aind] > L[j] / 2.0f)
			xo[aind] -= L[j];
		else if (xo[aind] - x[aind] < -L[j] / 2.0f)
			xo[aind] += L[j];

		
		x[aind] = 2.0f * b * x[aind] - a * xo[aind]
			+ b * delt2 / mass[itype] * f[aind]
			+ b * delt / ( 2.0f * mass[itype] ) * (new_noise + old_noise[aind]);


		xo[aind] = xtmp;
		old_noise[aind] = new_noise;


		if (x[ind * Dim + j] > L[j])
			x[ind * Dim + j] -= L[j];
		else if (x[ind * Dim + j] < 0.0f)
			x[ind * Dim + j] += L[j];

		disp_sq = abs(xo[ind * Dim + j] - x[ind * Dim + j]);
		if (disp_sq >Lh[j]){disp_sq = L[j] - disp_sq;}
		total_disp = disp_sq * disp_sq;
	}
	d_states[ind] = l_state;
}