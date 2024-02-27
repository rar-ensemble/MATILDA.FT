// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include <curand_kernel.h>
#include <curand.h>

#include "Measure_surface_tension.h"
#include "globals.h"

using namespace std;

SurfaceTension::~SurfaceTension(){return;}

SurfaceTension::SurfaceTension(istringstream& iss) : Measure(iss){
    if (Dim != 3){
        die("Surface tension measurement only implemented in 3D!");
    }
    st_x = (float*)calloc(ns * Dim, sizeof(float));

    GAMMA_FLAG = 1;

    readRequiredParameter(iss, delta);
    readRequiredParameter(iss, freq);

    float sqrt_delta = sqrt(delta);

    st_L[0] = (1.0f - sqrt_delta) * L[0];
    st_L[1] = (1.0f - sqrt_delta) * L[1];
    st_L[2] = (1.0f + delta) * L[2];

    // get scaling factors

	for (int j = 0; j < Dim; j++) {
		scales[j] = st_L[j] / L[j];
	}   
    

	st_gvol = 1.f;
	for (int j = 0; j < Dim; j++) {
		st_dx[j] = st_L[j] / float(Nx[j]);
		st_gvol *= st_dx[j];
	}
	
}

int SurfaceTension::LogCheck(){

    if (step%freq ==0 && step > 1){
        return 1;
    }
    else{
        return 0;
    }
}
void SurfaceTension::AddMeasure() {

    if(step%log_freq && step > 1){

        //scale coordinates

        for (int ind = 0; ind < ns;++ind){
            for (int j = 0; j < Dim; j++) {
                st_x[ind*Dim + j] = x[ind][j] * scales[j];
            }
        }

        // copy new x coordinates to device

        cudaMemcpy(d_x, st_x, ns * Dim * sizeof(float),
            cudaMemcpyHostToDevice);
    }


    //measure energy and pressure







    //cleanup

    //restore old positions

    cudaMemcpy(d_x, x, ns * Dim * sizeof(float),
    cudaMemcpyHostToDevice);

}
