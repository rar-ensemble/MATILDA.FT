// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


// Evaluates bonding forces.
// Currently harmonic and FENE bonds supported
__global__ void d_bonds( 
	float *d_f,					// [ns*Dim] particle forces
	const int* d_n_bonds, 		// [ns] number of bonds on each partic
	const int* d_bonded_to,		// [ns*MAXBONDS] id of bond partner
	const int* d_bond_type, 	// [ns*MAXBONDS] index of bond type
	const float* d_bond_req, 	// [nBondTypes] r_eq of bond
	const float* d_bond_k,		// [nBondTypes] force const of bond
	const int* d_bond_style,	// [nBondTypes] bond style (harmonic, FENE, etc)
	const float *d_x, 			// [ns*Dim] particle positions
	const float *L, 			// [Dim] box dimensions
	const float *Lh, 			// [Dim] half-box dimensions
	const int ns, 				// number of particles
	const int MAXBONDS, 		// MAX number of bonds per particle
	const int Dim				// System dimensionality
	) {


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
		id2 = d_bonded_to[ind * MAXBONDS + i];
		btyp = d_bond_type[ind * MAXBONDS + i];
				
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
__global__ void d_bondStressEnergy(
	float* d_e, 			// [ns] energy of each particle
	float* d_vir, 			// [ns*n_P_comps] pressure tensor terms
	const float* d_x, 		// [ns*Dim] particle positions
	const int* d_n_bonds,   // [ns] number of bonds on each partic
	const int* d_bonded_to,	// [ns*MAXBONDS] id of bond partner
	const int* d_bond_type, // [ns*MAXBONDS] index of bond type
	const float* d_bond_req,// [nBondTypes] r_eq of bond
	const float* d_bond_k,	// [nBondTypes] force const of bond
	const int* d_bond_style,// [nBondTypes] bond style (harmonic, FENE, etc)
	const float *L, 		// [Dim] box dimensions
	const float *Lh, 		// [Dim] half-box dimensions
	const int ns, 			// number of particles
	const int MAX_BONDS, 	// MAX number of bonds per particle
	const int n_P_comps,	// number of pressure tensor components
	const int Dim			// System dimensionality
	) {


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

		if (mdr2 > 1.0E-4f) {
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
					// printf("Did not implement the virial yet nor f = infinity!!!!!!!\n");
				}
			}
		} // if ( mdr2 > 1.0E-5 )
		else
			mf = 0.0f;


		// Store pressure tensor stuffs
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