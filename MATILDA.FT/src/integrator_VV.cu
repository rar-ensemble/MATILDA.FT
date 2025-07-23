// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "globals.h"
#include "integrator_VV.h"
#include <random>

using namespace std;

VV::~VV(){return;}

VV::VV(std::istringstream& iss) : Integrator(iss) {
    readRequiredParameter(iss,v_max);
    readRequiredParameter(iss,dist);
    d_TDISP.resize(ns);
    thrust::fill(d_TDISP.begin(),d_TDISP.end(),0.0);
}

void VV::Integrate_1(){


    if (MAX_DISP == -1.0){
        MAX_DISP = 0.0;
        thrust::fill(d_TDISP.begin(),d_TDISP.end(),0.0);
    }

    else{
        // float max_val = thrust::reduce(d_TDISP.begin(), d_TDISP.end(), 0, thrust::plus<float>());
        thrust::device_vector<float>::iterator iter = thrust::max_element(d_TDISP.begin(), d_TDISP.end());
        unsigned int position = iter - d_TDISP.begin();
        float max_val = *iter;
        // std::cout << "The maximum value is " << max_val <<  std::endl;
        MAX_DISP += max_val;
    }

    d_VV_integrator_1<<<group->GRID, group->BLOCK>>>(d_x, d_f, d_v,
        d_typ, d_mass, delt, d_L, d_Lh, group->d_index.data(),
        group->nsites, Dim, v_max, step, d_states, dist, d_TDISP.data());
}

void VV::Integrate_2(){
d_VV_integrator_2<<<group->GRID, group->BLOCK>>>(d_x, d_f, d_v,
    d_typ, d_mass, delt, d_L, d_Lh, group->d_index.data(),
    group->nsites, Dim);
}



// Device functions to integrate using the velocity-Verlet algo
// integrator_1 is the half step before teh force call, 
// integrator 2 is after the force call

__global__ void d_VV_integrator_1(float* x, float* f, float*v, int *typ, float *mass, float delt, 
	float *L, float *Lh, thrust::device_ptr<int> d_index, int ns, int D, float v_max, int step, curandState *d_states, int dist,
    thrust::device_ptr<float> d_TDISP){

	const int list_ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (list_ind >= ns)
		return;

	const int ind = d_index[list_ind];
	int itype = typ[ind];
    float disp_sq = 0.0;

    if (step == 1 && v_max != 0){
        if (dist == 0){
            for (int j = 0; j < D; j++){
                float vel;
                curandState l_state;
                l_state = d_states[ind];
                vel = v_max * curand_normal(&l_state);
                d_states[ind] = l_state;
                v[ind * D + j] = vel;
            }
        }
        if(dist == 1){
            for (int j = 0; j < D; j++){
                float vel;
                curandState l_state;
                l_state = d_states[ind];
                vel = v_max * 2.0 * (curand_uniform(&l_state) - 0.5);
                d_states[ind] = l_state;
                v[ind * D + j] = vel;
            }
        }
    }

	for (int j = 0; j < D; j++) {
		v[ind * D + j] = v[ind * D + j] + (f[ind * D +j]/ mass[itype] * delt/2.0);
		x[ind * D + j] = x[ind * D + j]	+ delt * v[ind * D + j];
        disp_sq += (delt * v[ind * D + j])*(delt * v[ind * D + j]);

		if (x[ind * D + j] > L[j])
			x[ind * D + j] -= L[j];
		else if (x[ind * D + j] < 0.0)
			x[ind * D + j] += L[j];
	}
    d_TDISP[list_ind] = d_TDISP[list_ind] + sqrt(disp_sq);
}

__global__ void d_VV_integrator_2(float* x, float* f, float*v, int *typ, float *mass, float delt, 
	float *L, float *Lh, thrust::device_ptr<int> d_index, int ns, int D ) {

	const int list_ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (list_ind >= ns)
		return;

	const int ind = d_index[list_ind];
	int itype = typ[ind];
	for (int j = 0; j < D; j++) 
		v[ind * D + j] = v[ind * D + j] + (f[ind * D +j]/ mass[itype] * delt/2.0); 
		
}
