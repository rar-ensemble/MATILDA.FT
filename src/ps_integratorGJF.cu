// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "ps_integratorGJF.h"
#include "PS_Box.h"

GJF::~GJF(){
    cudaFree(d_xOld);
    cudaFree(d_noiseOld);
    cudaFree(d_gjf_a);
    cudaFree(d_gjf_b);
    cudaFree(d_gjf_noiseMag);
    cudaFree(d_gjf_bdt2_over_m);
    cudaFree(d_gjf_bdt_over_2m);
}

GJF::GJF(std::istringstream& iss, PS_Box* box) : Integrator(iss, box) {

	int nDOF = mybox->nstot * mybox->returnDimension();

	cudaMalloc(&d_xOld, nDOF * sizeof(float));
	cudaMalloc(&d_noiseOld, nDOF * sizeof(float));

}

void GJF::Integrate_2(){

	int grid = mybox->psGroup[group_index].Grid;
	int block = mybox->psGroup[group_index].Block;


    d_GJF_integrator<<<grid, block>>>(mybox->d_x, d_xOld, d_noiseOld, mybox->d_f,
		d_gjf_a, d_gjf_b, d_gjf_noiseMag, d_gjf_bdt2_over_m, d_gjf_bdt_over_2m,
		mybox->d_intSpecies, mybox->d_L, mybox->psGroup[group_index].d_siteList,
		mybox->psGroup[group_index].nsites, mybox->returnDimension(), mybox->d_states);

}

// Initialization to be done after positions sent to
// the device in main init routines.
// Precomputes all per-type GJF coefficients so the kernel avoids
// divisions and sqrtf every step.
void GJF::finishInitialization() {

	int nDOF = mybox->nstot * mybox->returnDimension();

	// Initialize 'old' positions to current positions
	cudaMemcpy(d_xOld, mybox->d_x, nDOF*sizeof(float), cudaMemcpyDeviceToDevice);

	// Zero out the old noise values
	d_assignFloatVal<<<mybox->DnsGrid, mybox->nsBlock>>>(d_noiseOld, 0.0f, nDOF);

	delt2 = delt * delt;
	int nTypes = mybox->nTypes;
	float root2dt = sqrtf(2.0f * delt);

	float *h_a           = (float*) malloc(nTypes * sizeof(float));
	float *h_b           = (float*) malloc(nTypes * sizeof(float));
	float *h_noiseMag    = (float*) malloc(nTypes * sizeof(float));
	float *h_bdt2_over_m = (float*) malloc(nTypes * sizeof(float));
	float *h_bdt_over_2m = (float*) malloc(nTypes * sizeof(float));

	for (int i = 0; i < nTypes; i++) {
		float m  = mybox->speciesMass[i];
		float Di = mybox->speciesMobility[i];
		float b  = m / (m + delt / (2.0f * Di));
		float a  = (1.0f - delt / (2.0f * Di * m)) /
		           (1.0f + delt / (2.0f * Di * m));
		h_a[i]           = a;
		h_b[i]           = b;
		h_noiseMag[i]    = root2dt / sqrtf(Di);
		h_bdt2_over_m[i] = b * delt2 / m;
		h_bdt_over_2m[i] = b * delt / (2.0f * m);

		//std::cout << "GJF parameters type " << i << ": " << h_a[i] << " " << h_b[i] << " " << h_noiseMag[i] << " " << h_bdt2_over_m[i] << " " << h_bdt_over_2m[i] << std::endl;
	}

	cudaMalloc(&d_gjf_a,           nTypes * sizeof(float));
	cudaMalloc(&d_gjf_b,           nTypes * sizeof(float));
	cudaMalloc(&d_gjf_noiseMag,    nTypes * sizeof(float));
	cudaMalloc(&d_gjf_bdt2_over_m, nTypes * sizeof(float));
	cudaMalloc(&d_gjf_bdt_over_2m, nTypes * sizeof(float));

	cudaMemcpy(d_gjf_a,           h_a,           nTypes * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gjf_b,           h_b,           nTypes * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gjf_noiseMag,    h_noiseMag,    nTypes * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gjf_bdt2_over_m, h_bdt2_over_m, nTypes * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_gjf_bdt_over_2m, h_bdt_over_2m, nTypes * sizeof(float), cudaMemcpyHostToDevice);

	free(h_a);
	free(h_b);
	free(h_noiseMag);
	free(h_bdt2_over_m);
	free(h_bdt_over_2m);
}


// Device routine applying the Gronbech-Jensen/Farago integration scheme
// Comp Phys Comm V185 (2014) p524, Eqn 11.
// Per-type coefficients are precomputed in GJF::finishInitialization()
// to eliminate divisions and sqrtf from the hot path.
__global__ void d_GJF_integrator(
	float* x,                       // [nstot*Dim] positions to be updated
	float* xo,                      // [nstot*Dim] previous positions
	float* old_noise,               // [nstot*Dim] previous noise
	const float* f,                 // [nstot*Dim] force
	const float* gjf_a,             // [ntypes] GJF friction coefficient a
	const float* gjf_b,             // [ntypes] GJF friction coefficient b
	const float* gjf_noiseMag,      // [ntypes] sqrt(2*delt/D)
	const float* gjf_bdt2_over_m,   // [ntypes] b*delt^2/mass
	const float* gjf_bdt_over_2m,   // [ntypes] b*delt/(2*mass)
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

	int itype = typ[ind];

	// All per-type constants: 5 reads, zero arithmetic
	float b            = gjf_b[itype];
	float a            = gjf_a[itype];
	float noiseMag     = gjf_noiseMag[itype];
	float bdt2_over_m  = gjf_bdt2_over_m[itype];
	float bdt_over_2m  = gjf_bdt_over_2m[itype];

	for (int j = 0; j < Dim; j++) {
		int aind = ind * Dim + j;

		float new_noise = noiseMag * curand_normal(&l_state);

		float xtmp = x[aind];

		if (xo[aind] - x[aind] > L[j] / 2.0f)
			xo[aind] -= L[j];
		else if (xo[aind] - x[aind] < -L[j] / 2.0f)
			xo[aind] += L[j];

		x[aind] = 2.0f * b * x[aind] - a * xo[aind]
			+ bdt2_over_m * f[aind]
			+ bdt_over_2m * (new_noise + old_noise[aind]);

		xo[aind] = xtmp;
		old_noise[aind] = new_noise;

		if (x[aind] > L[j])
			x[aind] -= L[j];
		else if (x[aind] < 0.0f)
			x[aind] += L[j];
	}

	d_states[ind] = l_state;
}
