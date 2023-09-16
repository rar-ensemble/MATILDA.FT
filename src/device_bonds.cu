// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).



#include <stdio.h>
// Evaluates forces from u(rij) = k/2*(|rij|-req)^2;
__global__ void d_bonds(int* d_n_bonds, int* d_bonded_to,
	int* d_bond_type, float* d_bond_req, float* d_bond_k,
	float *d_x, float *d_f, float *L, float *Lh, 
	int ns, int MAX_BONDS, int Dim, int* d_bond_style) {


	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= ns)
		return;

	
	// Initialize variables
	float mdr2, mdr, mf;
	int i, j, id2, btyp; 
	float lforce[3], x1[3], dr[3] ;

	for (j = 0; j < Dim; j++) {
		lforce[j] = 0.0f;
		x1[j] = d_x[ind * Dim + j];
	}

	
	for (i = 0; i < d_n_bonds[ind]; i++) {
		id2 = d_bonded_to[ind * MAX_BONDS + i];
		btyp = d_bond_type[ind * MAX_BONDS + i];
		
		
		mdr2 = 0.0f;
		for (j = 0; j < Dim; j++) {
			dr[j] = x1[j] - d_x[id2 * Dim + j];
			
			if (dr[j] > Lh[j]) dr[j] -= L[j];
			else if (dr[j] < -Lh[j]) dr[j] += L[j];

			mdr2 += dr[j] * dr[j];
		}
		
		if (mdr2 > 1.0E-4f) {
			mdr = sqrtf(mdr2);
			// harmonic
			if (d_bond_style[btyp] == 0){
				mf = 2.0f * d_bond_k[btyp] * (mdr - d_bond_req[btyp]) / mdr;
			}
			// FENE
			else if(d_bond_style[btyp] == 1){
				mf = 2.0f * d_bond_k[btyp] * mdr/(1 - (mdr/d_bond_req[btyp]) * (mdr/d_bond_req[btyp])) / mdr;
			}
		}
		else
			mf = 0.0f;
		
		for (j = 0; j < Dim; j++) {
			lforce[j] -= mf * dr[j];
		}

	}// i=0 ; i<d_n_bonds[ind]
	

	for (j = 0; j < Dim; j++) {
		d_f[ind * Dim + j] += lforce[j];
	}
	
}

// Evaluates the per-particle energy, virial contributions
// these will be summed on the host
// NOTE: this double-counts the energy and virial, which will
// need to be corrected on the host.

__global__ void d_bondStressEnergy(int* d_n_bonds, int* d_bonded_to,
	int* d_bond_type, float* d_bond_req, float* d_bond_k,
	float* d_x, float* d_e, float* d_vir, float* L, float* Lh,
	int ns, int MAX_BONDS, int n_P_comps, int Dim, int* d_bond_style) {


	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= ns)
		return;


	// Initialize variables
	float mdr2, mdr, mf;
	int i, j, id2, btyp, bs;
	float x1[3], dr[3];

	for (j = 0; j < Dim; j++) {
		x1[j] = d_x[ind * Dim + j];
	}

	d_e[ind] = 0.f;
	for (i = 0; i < n_P_comps; i++)
		d_vir[ind * n_P_comps + i] = 0.f;


	for (i = 0; i < d_n_bonds[ind]; i++) {
		id2 = d_bonded_to[ind * MAX_BONDS + i];
		btyp = d_bond_type[ind * MAX_BONDS + i];
		bs = d_bond_style[btyp];


		mdr2 = 0.0f;
		for (j = 0; j < Dim; j++) {
			dr[j] = x1[j] - d_x[id2 * Dim + j];
			if (dr[j] > 0.5f * Lh[j]) dr[j] -= L[j];
			else if (dr[j] < -0.5f * Lh[j]) dr[j] += L[j];
			mdr2 += dr[j] * dr[j];
		}

		if (mdr2 > 1.0E-5f) {
			mdr = sqrtf(mdr2);

			if (bs == 0){
				float arg = (mdr - d_bond_req[btyp]);
				mf = 2.0f * d_bond_k[btyp] * arg / mdr;
				d_e[ind] += d_bond_k[btyp] * arg * arg;
			}
			else if (bs == 1){
				if ((mdr/d_bond_req[btyp]) < 1){
					float arg = 1/(1 - (mdr/d_bond_req[btyp]) * (mdr/d_bond_req[btyp]));
					mf = 2.0f * d_bond_k[btyp] * arg;
					d_e[ind] += mf * mdr;
					printf("Did not implement the virial yet nor f = infinity!!!!!!!\n");
				}
			}
		}
		else
			mf = 0.0f;

		if (bs == 0){
			d_vir[ind * n_P_comps + 0] += -mf * dr[0] * dr[0];
			d_vir[ind * n_P_comps + 1] += -mf * dr[1] * dr[1];
			if ( Dim == 2 )
				d_vir[ind * n_P_comps + 2] += -mf * dr[0] * dr[1];
			else if (Dim == 3) {
				d_vir[ind * n_P_comps + 2] += -mf * dr[2] * dr[2];
				d_vir[ind * n_P_comps + 3] += -mf * dr[0] * dr[1];
				d_vir[ind * n_P_comps + 4] += -mf * dr[0] * dr[2];
				d_vir[ind * n_P_comps + 5] += -mf * dr[1] * dr[2];
			}
		}

		else if(bs == 1){
			d_vir[ind * n_P_comps + 0] += -mf * dr[0] * dr[0];
			d_vir[ind * n_P_comps + 1] += -mf * dr[1] * dr[1];
			if ( Dim == 2 )
				d_vir[ind * n_P_comps + 2] += -mf * dr[0] * dr[1];
			else if (Dim == 3) {
				d_vir[ind * n_P_comps + 2] += -mf * dr[2] * dr[2];
				d_vir[ind * n_P_comps + 3] += -mf * dr[0] * dr[1];
				d_vir[ind * n_P_comps + 4] += -mf * dr[0] * dr[2];
				d_vir[ind * n_P_comps + 5] += -mf * dr[1] * dr[2];
			}
		}
	}
}