// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include <curand_kernel.h>
#include <curand.h>

#include "Measure_surface_tension.h"
#include "globals.h"

using namespace std;

SurfaceTension::~SurfaceTension(){return;}

SurfaceTension::SurfaceTension(istringstream& iss) : Measure(iss){


    SurfaceTensions.push_back(this);

    gvol0 = gvol;

    if (Dim != 3){
        die("Surface tension measurement only implemented in 3D!");
    }

    p_x = (float*)calloc(ns * Dim, sizeof(float));
    p_x0 = (float*)calloc(ns * Dim, sizeof(float));

    GAMMA_FLAG = 1;

    readRequiredParameter(iss, delta);
    readRequiredParameter(iss, freq);

    float sqrt_delta = sqrt(delta);

    p_L[0] = (1.0f - sqrt_delta) * L[0];
    p_L[1] = (1.0f - sqrt_delta) * L[1];
    p_L[2] = (1.0f + delta) * L[2];

    // get scaling factors

	for (int j = 0; j < Dim; j++) {
		scales[j] = p_L[j] / L[j];
	}   
    

	p_gvol = 1.f;
	for (int j = 0; j < Dim; j++) {
		p_dx[j] = p_L[j] / float(Nx[j]);
		p_gvol *= p_dx[j];
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
void SurfaceTension::PerturbState() {

        //scale coordinates



        cudaMemcpy(p_x0, d_x, ns * Dim * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_ns_float, d_x, ns * Dim * sizeof(float), cudaMemcpyDeviceToHost);
        for (int i = 0; i < ns; i++)
            for (int j = 0; j < Dim; j++)
               p_x[i*Dim + j] = h_ns_float[i * Dim + j];

        // copy new x coordinates to device

        cudaMemcpy(d_x, p_x, ns * Dim * sizeof(float),cudaMemcpyHostToDevice);

        cudaMemcpy(d_L, p_L, Dim * sizeof(float),cudaMemcpyHostToDevice);

        cudaMemcpy(d_dx, p_dx, Dim * sizeof(float),cudaMemcpyHostToDevice);

        gvol = p_gvol;

}



void SurfaceTension::RestoreState(){

    //restore old positions
    cudaMemcpy(d_dx, p_x0, ns * Dim, cudaMemcpyHostToDevice);

    cudaMemcpy(d_L, L, Dim * sizeof(float), cudaMemcpyHostToDevice);

    cudaMemcpy(d_dx, dx, Dim * sizeof(float),cudaMemcpyHostToDevice);

    gvol = gvol0;
}

