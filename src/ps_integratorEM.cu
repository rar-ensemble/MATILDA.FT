// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "ps_integratorEM.h"
#include "PS_Box.h"

EM::~EM(){

}

EM::EM(std::istringstream& iss, PS_Box* box) : Integrator(iss, box) {

	int nDOF = mybox->nstot * mybox->returnDimension();

}

void EM::Integrate_2(){

	int grid = mybox->psGroup[group_index].Grid;
	int block = mybox->psGroup[group_index].Block;

    d_EM_integrator<<<grid, block>>>(mybox->d_x, mybox->d_f,
		delt, sqrtf(2.0 * delt), mybox->d_intSpecies, mybox->d_L, 
		mybox->psGroup[group_index].d_siteList, mybox->psGroup[group_index].nsites,
		mybox->returnDimension(), mybox->d_states);

}

// Initialization to be done after positions sent to
// the device in main init routines.
void EM::finishInitialization() {



}



__global__ void d_EM_integrator(
	float* x,                       // [nstot*Dim] positions to be updated
	const float* f,                 // [nstot*Dim] force
	const float delt,				// time step magnitude
	const float noiseMag,			// sqrt(2.0 * delt)
	const int* typ,                 // [nstot] particle type index
	const float* L,                 // [Dim] box dimensions
	const int* d_index,             // [ns] site list for this integrator group
	const int ns,                   // number of sites in this group
	const int Dim,                  // system dimensionality
	curandState* d_states           // [nstot] RNG state
	) {

	int list_ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (list_ind >= ns)
		return;

	int ind = d_index[list_ind];

	curandState l_state = d_states[ind];

	// int itype = typ[ind];


	for (int j = 0; j < Dim; j++) {
		int aind = ind * Dim + j;

		float new_noise = noiseMag * curand_normal(&l_state);

		float xtmp = x[aind];

		xtmp = xtmp + delt * f[aind] + new_noise;


		if (xtmp > L[j])
			xtmp -= L[j];
		else if (xtmp < 0.0f)
			xtmp += L[j];

		x[aind] = xtmp;
	}

	d_states[ind] = l_state;
}
