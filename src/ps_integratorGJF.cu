// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "ps_integratorGJF.h"
#include "PS_Box.h"

GJF::~GJF(){return;}

GJF::GJF(std::istringstream& iss, PS_Box* box) : Integrator(iss, box) {
    
    using_GJF = 1;

	int nDOF = mybox->nstot * mybox->returnDimension();

	cudaMalloc(&d_xOld, nDOF * sizeof(float));
	cudaMalloc(&d_noiseOld, nDOF * sizeof(float));

}

void GJF::Integrate_2(){

	int grid = mybox->psGroup[group_index].Grid;
	int block = mybox->psGroup[group_index].Block;

    d_GJF_integrator<<<grid, block>>>(mybox->d_x, d_xOld, d_noiseOld, mybox->d_f, 
		mybox->d_speciesMass, mybox->d_speciesMobility, mybox->d_intSpecies,
		delt, sqrtf(2.0*delt), mybox->d_L, mybox->psGroup[group_index].d_siteList,
		mybox->psGroup[group_index].nsites, mybox->returnDimension(), mybox->d_states);

}

// Initialization to be done after positions sent to 
// the device in main init routines
void GJF::finishInitialization() {
	
	int nDOF = mybox->nstot * mybox->returnDimension();
	
	// Initialize 'old' positions current positions
	cudaMemcpy(d_xOld, mybox->d_x, nDOF*sizeof(float), cudaMemcpyHostToDevice);

	// Zero out the old noise values
	d_assignFloatVal<<<mybox->DnsGrid, mybox->nsBlock>>>(d_noiseOld, 0.0f, nDOF);

}

// device routine that applies the Stormer-Verlet Gronbech-Jensen and 
// Farago integration scheme from Comp PHys Comm V185 (2014) p524, Eqn 11.
// Notation mostly follows GJF with diff = 1/alpha, where 'alpha' is the friction
// coefficient defined in their paper below eqn 1.
__global__ void d_GJF_integrator(
	float* x, 				// [nstot*Dim] positiosn to be updated
	float* xo, 				// [nstot*Dim] previous positions
	float* old_noise,		// [nstot*Dim] previous noise
	const float* f, 		// [nstot*Dim] force
	const float *mass, 		// [ntypes] masses by particle type
	const float *diff, 		// [ntypes] mobility/diffusivity by particle type
	const int *typ, 		// [nstot] particle types
	const float delt, 		// size of time step
	const float root2dt, 	// part of noise magnitude, sqrt(2*delt)
	const float* L, 		// [Dim] box dimensions
	const int* d_index,		// [ns] list of indices for particles in this group
	const int ns, 			// number of sites in group to be integrated (ns <= nstot)
	const int Dim, 			// system dimensionality
	curandState* d_states	// [nstot*Dim] RNG state variable
	) {

	int list_ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (list_ind >= ns)
		return;
		
	int ind = d_index[list_ind];
		
	curandState l_state;
	l_state = d_states[ind];

	int itype = typ[ind];

	float b = mass[itype] / (mass[itype] + delt / (2.0f * diff[itype]));
	
	float a = (1.0f - delt / (2.0f * diff[itype] * mass[itype])) /
		(1.0f + delt / (2.0f * diff[itype] * mass[itype]));
	
	float delt2 = delt * delt;

	float noiseMag = root2dt / sqrtf(diff[itype]);

	for (int j = 0; j < Dim; j++) {
		int aind = ind * Dim + j;

		float new_noise = noiseMag * curand_normal(&l_state);
		
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


		if (x[aind] > L[j])
			x[aind] -= L[j];
		else if (x[aind] < 0.0f)
			x[aind] += L[j];

	}

	d_states[ind] = l_state;
}