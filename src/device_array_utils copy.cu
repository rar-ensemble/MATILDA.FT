// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#define PI 3.141592654f
#define PI2 6.2831853071f
#include <cmath>
#include <cufft.h>
#include <cufftXt.h>

// receives id \in [0,M) and fills array nn[Dim] with values
// nn[i] \in [0, Nx[i])
__device__ void d_unstack(int id, int *nn, 
	const int *NX, const int d) {
	if (d == 1) {
		nn[0] = id;
		return;
	}
	else if (d == 2) {
		nn[1] = id / NX[0];
		//nn[0] = (id - nn[1] * NX[0]);
		nn[0] = id % NX[0];
		return;
	}
	else if (d == 3) {
		int idx = id;
		nn[2] = idx / (NX[1] * NX[0]);
		idx -= (nn[2] * NX[0] * NX[1]);
		nn[1] = idx / NX[0];
		nn[0] = idx % NX[0];
		/*nn[2] = id / NX[1] / NX[2];
		nn[1] = id / NX[0] - nn[2] / NX[1];
		nn[0] = id - (nn[1] + nn[2] * NX[1]) * NX[0];*/
		return;
	}
}

__device__ void d_get_r(const int id, float r[3], const int* Nx, 
	const float* dx, const int Dim) {
	int nn[3];
	d_unstack(id, nn, Nx, Dim);

	for (int i = 0; i < Dim; i++)
		r[i] = float(nn[i]) * dx[i];
}


__device__ float d_pbc_mdr2(float r1[3], float r2[3], float dr[3],
	const float L[3], const float Lh[3], const int Dim) {

	float mdr2 = 0.0f;
	for (int j = 0; j < Dim; j++) {
		dr[j] = r1[j] - r2[j];
		if (dr[j] > Lh[j]) dr[j] -= L[j];
		else if (dr[j] < -Lh[j]) dr[j] += L[j];

		mdr2 += dr[j] * dr[j];
	}

	return mdr2;
}


__device__ float d_get_k(int id, float k[3], const float L[3],
	const int Nx[3], const int Dim) {
	float k2 = 0.0f;
	int nn[3];

	d_unstack(id, nn, Nx, Dim);
	
	for (int i = 0; i < Dim; i++) {
		if (float(nn[i]) < float(Nx[i]) / 2.0)
			k[i] = PI2 * float(nn[i]) / L[i];
		else
			k[i] = PI2 * float(nn[i] - Nx[i]) / L[i];
		k2 += k[i] * k[i];
	}
	
	/*
	for (int i = 0; i < Dim; i++) {
		k[i] = PI2 * float(nn[i]) / L[i];
		k2 += k[i] * k[i];
	}
	*/
	return k2;
}







__global__ void d_make_dens_step(float* rh_all, float* dL,
	float* d_dx, int* dNx, int Dim, int M, int ntypes) {

	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= M)
		return;

	float r[3];
	d_get_r(ind, r, dNx, d_dx, Dim);
	
	float step = 0.5 * (1.0 - tanhf((fabsf(r[1] - dL[1]/2.0f) - dL[1] / 4.0) / 1.0));
	rh_all[ind] = step;
	if (ntypes > 1)
		rh_all[M + ind] = 1.0f - step; 
}

__global__ void d_make_step(cufftComplex* tmp, float* dL, 
	float* d_dx, int* dNx, int Dim, int M) {

	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= M)
		return;

	float r[3];
	d_get_r(ind, r, dNx, d_dx, Dim);

	tmp[ind].y = 0.0f;

	if (r[0] < dL[0] / 4.0f ||
		r[0] > 3.0f * dL[0] / 4.0f)
		tmp[ind].x = 0.0f;
	else
		tmp[ind].x = 1.0f;

}



// puts the real field tp into the complex field cpx1, 
// setting imag(cpx1) = 0
__global__ void d_real2complex(float* tp, cufftComplex* cpx1, int M) {
	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= M)
		return;

	cpx1[ind].x = tp[ind];
	cpx1[ind].y = 0.0f;
}

// Assumes the imaginary part of cp1 is zero, which is ignored
__global__ void d_complex2real(cufftComplex* cp1, float* tp, int M) {
	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= M)
		return;

	tp[ind] = cp1[ind].x;
}

__global__ void d_multiply_cufftCpx_scalar(cufftComplex* vec, float scalar, int max) {
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= max)
		return;

	vec[id].x *= scalar;
	vec[id].y *= scalar;
}

__global__ void d_multiply_cufftCpx_scalar(cufftComplex* in, float scalar, cufftComplex* out, int max) {
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= max)
		return;

	out[id].x = in[id].x * scalar;
	out[id].y = in[id].y * scalar;
}

__global__ void d_multiply_float_scalar(float* vec, float scalar, int max) {
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= max)
		return;

	vec[id] *= scalar;
}

__global__ void d_multiply_float_scalar(float* in, float scalar, float* out, int max) {
	const int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= max)
		return;

	out[id] = in[id] * scalar;
}

// Puts density field of type typ into real part of d_tp,
// sets imag(d_tp) = 0
__global__ void d_prepareDensity(int typ, float* d_t_rho, 
	cufftComplex* d_tp, int M) {

	const int ind = blockIdx.x * blockDim.x + threadIdx.x;

	if (ind >= M)
		return;

	d_tp[ind].x = d_t_rho[typ * M + ind];
	d_tp[ind].y = 0.f;
}




// Puts charge density field into real part of d_tc 
// sets imaginary part of d_tc = 0
__global__ void d_prepareChargeDensity(float* d_t_charge_density,
	cufftComplex* d_tc, int M) {

	const int ind = blockIdx.x * blockDim.x + threadIdx.x;

    if (ind >= M)
	    return;

	d_tc[ind].x = d_t_charge_density[ind];
	d_tc[ind].y = 0.f;
}


//calculates electrostatic potential in Fourier space
// d_tc:         [M] Fourier transform of density field
// d_ep:         [M] variable to store electrostatic potential
// bjerrum:      Bjerrum length
// length_scale: charge smearing length

__global__ void d_prepareElectrostaticPotential(cufftComplex* d_tc, cufftComplex* d_ep, 
	float bjerrum, float length_scale, const int M, const int Dim, const float* L,
	const int* Nx) {
	
	const int ind = blockIdx.x * blockDim.x + threadIdx.x;

	if (ind >= M)
		return;

	float kv[3], k2;
	k2 = d_get_k(ind, kv, L, Nx, Dim);

	if (k2 != 0) {
		//d_ep[ind].x = ((d_tc[ind].x * 4 * PI * bjerrum) / k2) * exp(-1 * k2 / (2 * length_scale * length_scale));
		d_ep[ind].x = ((d_tc[ind].x * 4 * PI * bjerrum) / k2) * exp( -k2 * length_scale * length_scale / 2.0);
		d_ep[ind].y = ((d_tc[ind].y * 4 * PI * bjerrum) / k2) * exp( -k2 * length_scale * length_scale / 2.0);
	}
	else {
		d_ep[ind].x = 0.f;
		d_ep[ind].y = 0.f;
	}
}

//calculates electric field
// d_ep = -grad d_ef
// In k-space: d_ep = -i*k*d_ef

// __global__ void d_prepareElectricField(cufftComplex* d_ef, cufftComplex* d_ep,
// 	float length_scale, const int M, const int Dim, const float* L,
// 	const int* Nx, const int dir) {
	
// 	const int ind = blockIdx.x * blockDim.x + threadIdx.x;

// 	if (ind >= M)
// 		return;

// 	float k[3], k2;
// 	k2 = d_get_k(ind, k, L, Nx, Dim);

// 	d_ef[ind].x = k[dir] * d_ep[ind].y * exp(-k2 * length_scale * length_scale /2.0);
// 	d_ef[ind].y = -k[dir] * d_ep[ind].x * exp( -k2 * length_scale * length_scale / 2.0);
// 	//d_ef[ind].x = k[dir] * d_ep[ind].x * exp(-1 * k2 / (2 * length_scale * length_scale));
// 	//d_ef[ind].y = 0.f;
// }


//calculates electric field
// d_ep = -grad d_ef
// In k-space: d_ep = -i*k*d_ef
__global__ void d_prepareElectricField(cufftComplex* d_cpxx, cufftComplex* d_cpxy,cufftComplex* d_cpxz, cufftComplex* d_ep,
	float length_scale, const int M, const int Dim, const float* L,
	const int* Nx) {
	
	const int ind = blockIdx.x * blockDim.x + threadIdx.x;

	if (ind >= M)
		return;

	float k[3], k2;
	int dir;
	k2 = d_get_k(ind, k, L, Nx, Dim);

	float exp_k = exp(-k2 * length_scale * length_scale /2.0);
	float d_ep_y = d_ep[ind].y;
	float d_ep_x = d_ep[ind].x;

	dir = 0; //z

	d_cpxx[ind].x = k[dir] * d_ep_y * exp_k;
	d_cpxx[ind].y = -k[dir] * d_ep_x * exp_k;

	dir = 1; //z

	d_cpxy[ind].x = k[dir] * d_ep_y * exp_k;
	d_cpxy[ind].y = -k[dir] * d_ep_x * exp_k;

	if (Dim == 3){

		dir = 2; //z

		d_cpxz[ind].x = k[dir] * d_ep_y * exp_k;
		d_cpxz[ind].y = -k[dir] * d_ep_x * exp_k;

	}

	//d_ef[ind].x = k[dir] * d_ep[ind].x * exp(-1 * k2 / (2 * length_scale * length_scale));
	//d_ef[ind].y = 0.f;
}

__global__ void d_setElectrostaticPotential(cufftComplex* d_ep,
	float* d_electrostatic_potential, const int M) {

	const int ind = blockIdx.x * blockDim.x + threadIdx.x;

	if (ind >= M)
		return;

	d_electrostatic_potential[ind] = d_ep[ind].x;
}

__global__ void d_setElectricField(cufftComplex* d_ef,
	float* d_electric_field, int dim, const int M) {

	const int ind = blockIdx.x * blockDim.x + threadIdx.x;

	if (ind >= M)
		return;

	d_electric_field[(M * dim) + ind] = d_ef[ind].x;
}


__global__ void d_resetComplexes(cufftComplex* one, cufftComplex* two,
	const int M) {

	const int ind = blockIdx.x * blockDim.x + threadIdx.x;

	if (ind >= M)
		return;

	one[ind].x = 0.f;
	one[ind].y = 0.f;

	two[ind].x = 0.f;
	two[ind].y = 0.f;
}




// c1 is complex-format result of the inverse Fourier transform 
__global__ void d_prepareIntegrand(cufftComplex* c1, int typ, float* d_t_rho,
	float* t_tp, int M) {

	const int ind = blockIdx.x * blockDim.x + threadIdx.x;

	if (ind >= M)
		return;

	t_tp[ind] = c1[ind].x * d_t_rho[typ * M + ind];

}


__global__ void d_prepareForceKSpace(cufftComplex* fk, cufftComplex* rhok,
	cufftComplex* rhofk, const int dir, const int Dim, const int M) {
	
	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= M)
		return;

	// Divides by float(M) to normalize the FFT
	rhofk[ind].x = (fk[ind * Dim + dir].x * rhok[ind].x
		- fk[ind * Dim + dir].y * rhok[ind].y) / float(M);

	rhofk[ind].y = (fk[ind * Dim + dir].x * rhok[ind].y 
		+ fk[ind * Dim + dir].y * rhok[ind].x) / float(M);

}



__global__ void d_prepareForceKSpace3D(cufftComplex* fk, cufftComplex* rhok,
	cufftComplex* rhofkx, cufftComplex* rhofky, cufftComplex* rhofkz,
	const int Dim, const int M) {
	
	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= M)
		return;

	// Divides by float(M) to normalize the FFT
	// rhofk[ind].x = (fk[ind * Dim + dir].x * rhok[ind].x- fk[ind * Dim + dir].y * rhok[ind].y) / float(M);
	// rhofk[ind].y = (fk[ind * Dim + dir].x * rhok[ind].y + fk[ind * Dim + dir].y * rhok[ind].x) / float(M);

	float rhok_x = rhok[ind].x;
	float rhok_y = rhok[ind].y;
	float fk_y;
	float fk_x;

	float f_M = float(M);
	int dir;

	// Divides by float(M) to normalize the FFT
	dir = 0;

	fk_y = fk[ind * Dim + dir].y;
	fk_x = fk[ind * Dim + dir].x;

	rhofkx[ind].x = (fk_x * rhok_x- fk_y * rhok_y) / f_M;
	rhofkx[ind].y = (fk_x * rhok_y + fk_y * rhok_x) / f_M;

//y
	dir = 1;

	fk_y = fk[ind * Dim + dir].y;
	fk_x = fk[ind * Dim + dir].x;

	rhofkx[ind].x = (fk_x * rhok_x- fk_y * rhok_y) / f_M;
	rhofkx[ind].y = (fk_x * rhok_y + fk_y * rhok_x) / f_M;
//z
	if (Dim == 3){
		dir = 2;

		fk_y = fk[ind * Dim + dir].y;
		fk_x = fk[ind * Dim + dir].x;

		rhofkx[ind].x = (fk_x * rhok_x- fk_y * rhok_y) / f_M;
		rhofkx[ind].y = (fk_x * rhok_y + fk_y * rhok_x) / f_M;
	}
}


// __global__ void d_prepareForceKSpace(cufftComplex* fk, cufftComplex* rhok, cufftComplex* rhofkx, cufftComplex* rhofky, cufftComplex* rhofkz, const int Dim, const int M) { 
	
// 	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (ind >= M)
// 		return;

// 	float rhok_x = rhok[ind].x;
// 	float rhok_y = rhok[ind].y;

// 	float f_M = float(M);

// 	int dir;

// 	// Divides by float(M) to normalize the FFT
// 	dir = 0;

// 	float fk_y = fk[ind * Dim + dir].y;
// 	float fk_x = fk[ind * Dim + dir].x;

// 	rhofkx[ind].x = (fk_x * rhok_x
// 		- fk_y * rhok_y) / f_M;

// 	rhofkx[ind].y = (fk_x * rhok_y 
// 		+ fk_y * rhok_x) / f_M;

// //y
// 	dir = 1;

// 	fk_y = fk[ind * Dim + dir].y;
// 	fk_x = fk[ind * Dim + dir].x;

// 	rhofkx[ind].x = (fk_x * rhok_x
// 		- fk_y * rhok_y) / f_M;

// 	rhofkx[ind].y = (fk_x * rhok_y 
// 		+ fk_y * rhok_x) / f_M;
// //z
// 	if (Dim == 3){
// 		dir = 2;

// 		fk_y = fk[ind * Dim + dir].y;
// 		fk_x = fk[ind * Dim + dir].x;

// 		rhofkx[ind].x = (fk_x * rhok_x
// 			- fk_y * rhok_y) / f_M;

// 		rhofkx[ind].y = (fk_x * rhok_y 
// 			+ fk_y * rhok_x) / f_M;
// 	}
// }

__global__ void d_multiplyComplex(cufftComplex* uk, cufftComplex* rhok,
	cufftComplex* rhouk, const int M) {
	
	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= M)
		return;

	rhouk[ind].x = (uk[ind].x * rhok[ind].x
					- uk[ind].y * rhok[ind].y) / float(M);

	rhouk[ind].y = (uk[ind].x * rhok[ind].y
					+ uk[ind].y * rhok[ind].x) / float(M);
}


__global__ void d_divideByDimension(cufftComplex* in, const int M) {
	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= M)
		return;

	in[ind].x *= 1.0f / float(M);
	in[ind].y *= 1.0f / float(M); 
}


// Takes u(k) and multiplies by I*k. 
__global__ void d_kSpaceMakeForce(cufftComplex* u_k, cufftComplex* f_k,
	const float *L, const int *Nx, const int M, const int Dim) {
	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= M)
		return;

	float kv[3];
	(void)d_get_k(ind, kv, L, Nx, Dim);

	f_k[ind * Dim + 0].x = -kv[0] * u_k[ind].y;
	f_k[ind * Dim + 0].y = -kv[0] * u_k[ind].x;

	f_k[ind * Dim + 1].x = -kv[1] * u_k[ind].y;
	f_k[ind * Dim + 1].y = -kv[1] * u_k[ind].x;

	if (Dim == 3) {
		f_k[ind * Dim + 2].x = -kv[2] * u_k[ind].y;
		f_k[ind * Dim + 2].y = -kv[2] * u_k[ind].x;
	}

}

__global__ void d_accumulateGridForce(cufftComplex* frho,
	float* d_t_rho, float* d_t_fdir, const int type, const int M) {

	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= M)
		return;

	d_t_fdir[type * M + ind] = d_t_fdir[type * M + ind] +
		d_t_rho[type * M + ind] * frho[ind].x;

}

__global__ void d_accumulateGridForceWithCharges(cufftComplex* f_electric_field,
	float* d_t_charge_density, float* d_t_fdir, const int M) {

	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= M)
		return;

	d_t_fdir[ind] = f_electric_field[ind].x;
	//d_t_fdir[ind] = d_t_fdir[ind] + (d_t_charge_density[ind] * f_electric_field[ind].y);
}

// fk[i*Dim+dir] = tp[i]
// Assumes fk real, tp complex
__global__ void d_insertForceCompC2R(float* f,
	cufftComplex* tp, const int dir, const int Dim, const int M) {

	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= M)
		return;

	f[ind * Dim + dir] = tp[ind].x;

}

// tp[i] = fk[i*Dim+dir]
__global__ void d_extractForceComp(cufftComplex* tp,
	cufftComplex* fk, const int dir, const int Dim, const int M) {
	
	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= M)
		return;

	tp[ind] = fk[ind * Dim + dir];

}

__global__ void d_prepareFieldForce(cufftComplex* frhoA, cufftComplex* frhoB,
	const float* d_t_rho, const float* d_t_fdir, 
	const int typeA, const int typeB, 
	const int dir, const int Dim, const int M) {

	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= M)
		return;

	frhoA[ind].y = frhoB[ind].y = 0.f;

	frhoA[ind].x =  d_t_rho[typeA * M + ind] * d_t_fdir[ind * Dim + dir];
	frhoB[ind].x = -d_t_rho[typeB * M + ind] * d_t_fdir[ind * Dim + dir];
}



__global__ void d_complex2real(cufftComplex* cpx1, float* tp, float* tp2, int M) {
	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= M)
		return;

	tp[ind] = cpx1[ind].x;
	tp2[ind] = cpx1[ind].y;

}

__global__ void d_assignValueR(float* vec, float val, int Dim, int ns) {
	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= ns)
		return;

	for (int j = 0; j < Dim; j++)
		vec[ind * Dim + j] = val;

}

__global__ void d_copyPositions(float* xdest, float* xorig, int Dim, int ns) {
	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= ns)
		return;

	for (int j = 0; j < Dim; j++)
		xdest[ind * Dim + j] = xorig[ind * Dim + j];
}

__global__ void d_initVirial(float* vr, const float* fr, 
	const float* dL, const float* Lh, const float* dx, const int Dim, 
	const int n_P_comps, const int* Nx, const int M) {

	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= M)
		return;

	float ro[3], ri[3], dr[3];

	for (int j = 0; j < Dim; j++) {
		ro[j] = 0.f;
	}

	d_get_r(ind, ri, Nx, dx, Dim);

	(void)d_pbc_mdr2(ri, ro, dr, dL, Lh, Dim);

	for (int j = 0; j < Dim; j++) {
		// Whether Dim == 2 or 3, this provides the diagonal component

		vr[ind * n_P_comps + j] = dr[j] * fr[ind * Dim + j];


		if (Dim == 2 && j == 0) {
			vr[ind * n_P_comps + 2] = dr[1] * fr[ind * Dim + j]; // XY
		}

		if (Dim == 3 && j == 0) {
			vr[ind * n_P_comps + 3] = dr[1] * fr[ind * Dim + j]; // XY
			vr[ind * n_P_comps + 4] = dr[2] * fr[ind * Dim + j]; // XZ
		}

		if (Dim == 3 && j == 1) {
			vr[ind * n_P_comps + 5] = dr[2] * fr[ind * Dim + j]; // YZ
		}
	}

}

__global__ void d_prepareVirial(const cufftComplex* vr,
	const cufftComplex* rh1, cufftComplex* out,
	const int Pterm, const int Dim, const int n_P_comps, const int M)
{
	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= M)
		return;

	int pind = ind * n_P_comps + Pterm;
	out[ind].x = vr[pind].x * rh1[ind].x - vr[pind].y * rh1[ind].y;
	out[ind].y = vr[pind].y * rh1[ind].x + vr[pind].x * rh1[ind].y;
}

__global__ void d_extractVirialCompR2C(cufftComplex* ot, const float* vin, const int comp,
	const int n_P_comps, const int M) {

	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= M)
		return;

	ot[ind].y = 0.f;

	ot[ind].x = vin[ind * n_P_comps + comp];

}


__global__ void d_insertVirialCompC2C(cufftComplex* ot, const cufftComplex* vin, const int comp,
	const int n_P_comps, const int M) {

	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= M)
		return;

	ot[ind * n_P_comps + comp].x = vin[ind].x;
	ot[ind * n_P_comps + comp].y = vin[ind].y;

}



