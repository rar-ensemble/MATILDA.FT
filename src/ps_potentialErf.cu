// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "ps_potentialErf.h"
#include "PS_Box.h"



NBErf::NBErf() {}
NBErf::~NBErf() {}

// Constructor called by the "factor" routine in ps_potential.cu
NBErf::NBErf(std::istringstream& iss, PS_Box* box) : PS_Potential(iss, box) {
    
    iss >> grpI;
    iss >> grpJ;

    iss >> Ao;
    iss >> Rp;
    iss >> sigma;
}

void NBErf::initializePotential() {
    std::cout << "Initializing Erf potential..." << std::endl;

    PS_Potential::initializePotential();
    
    std::complex<float> I(0.0, 1.0);
    float kv[3], k2, kmag, Rp3;
    int Dim = mybox->returnDimension();
    int M = mybox->M;
    Rp3 = Rp * Rp * Rp;
    
    float r0[3], ri[3], dr[3];
    r0[0] = r0[1] = r0[2] = 0.0f;

    for ( int i=0 ; i<M ; i++ ) {
        
        mybox->get_rf(i, ri);

        float mdr2 = mybox->pbc_dr2(dr, ri, r0);
        float mdr = sqrtf(mdr2);

        // Multiplication by V ensures proper normalization when used in
        // Fourier space.
        // Sqrt(2) * sigma term comes from 3D convolution of spherical step
        // func with Gaussian.
        ur[i] = Ao * mybox->V * (1.0 - erf((mdr - Rp)/(pow(2.0,0.5) * sigma)));



        // k2 = mybox->get_kD(i, kv);
        // kmag = sqrtf(k2);
        
        // if ( i == 0 ) {
        //     uk[i] = Ao * PI4 * Rp3 / 3.0;
        // }
        // else {
        //     uk[i] = Ao * exp(-k2 * sigma*sigma / 2.0) *
        //             PI4 * (sin(Rp*kmag) - Rp*kmag * cos(Rp*kmag) ) / k2 / kmag;
        // }
        
        // // In k-space, f(k) = -I * k * u(k)
        // for (int j = 0; j < Dim; j++) {
        //     fk[i * Dim + j] = -I * kv[j] * uk[i];
        // }
    }

    // Send these to device, inv transform to get ur, f(r)
    cudaMemcpy(d_uk, uk, M*sizeof(std::complex<float>), cudaMemcpyHostToDevice);
    
    // Compute inverse fft of d_uk, store in temp variable
    mybox->cufftWrapperSingle(d_uk, mybox->d_cpxAlex, -1);


    // d_ur = Real(d_cpxAlex)
    d_cpxToFloat<<<mybox->M_Grid, mybox->M_Block>>>(d_ur, mybox->d_cpxAlex, M);
    check_cudaError("cpxToFloat in Gaussian potential");

    // Same thing for force arrays
    cudaMemcpy(d_fk, fk, M*Dim*sizeof(std::complex<float>), cudaMemcpyHostToDevice);
    check_cudaError("sending force array to device in Gauss potential");


    // Copy potential, force functions back to host
    // only really used for debugging
    // ur = d_ur
    cudaMemcpy(ur, d_ur, M*sizeof(float), cudaMemcpyDeviceToHost);

}//initializePotential()