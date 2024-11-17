// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "ps_potentialGaussian.h"
#include "PS_Box.h"



NBGauss::NBGauss() {}
NBGauss::~NBGauss() {}

// Constructor called by the "factor" routine in ps_potential.cu
NBGauss::NBGauss(std::istringstream& iss, PS_Box* box) : PS_Potential(iss, box) {
    
    iss >> grpI;
    iss >> grpJ;

    iss >> Ao;

    float sigma;
    iss >> sigma;
    sig2 = sigma*sigma;
}

void NBGauss::initializePotential() {
    std::cout << "Initializing Gaussian potential..." << std::endl;

    Iind = mybox->findGroupInteger(grpI);
    Jind = mybox->findGroupInteger(grpJ);

    // Allocate grid force memory for groups I, J if needed
    if ( mybox->psGroup[Iind].hasForce() == 0 ) { mybox->psGroup[Iind].enableForce(); }
    if ( mybox->psGroup[Jind].hasForce() == 0 ) { mybox->psGroup[Jind].enableForce(); }
    
    std::complex<float> I(0.0, 1.0);
    float kv[3], k2;
    int Dim = mybox->returnDimension();
    int M = mybox->M;

    // uk: std::complex<float> [M]
    // fk: std::complex<float> [Dim*M]
    for ( int i=0 ; i<M ; i++ ) {
        k2 = mybox->get_kD(i, kv);
        uk[i] = Ao * exp(-k2 * sig2 / 2.0f) ;
        
        // In k-space, f(k) = -I * k * u(k)
        for (int j = 0; j < Dim; j++) {
            fk[i * Dim + j] = -I * kv[j] * uk[i];
        }
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
    cudaMemcpy(ur, d_ur, M*sizeof(float), cudaMemcpyDeviceToHost);




    // for ( int j=0 ; j<Dim ; j++ ) {
    //     d_extractCpxDirToCpx<<<mybox->M_Grid, mybox->M_Block>>>(mybox->d_cpxGabe, d_fk, j, Dim, M);
    //     check_cudaError("extracting component");

    //     mybox->cufftWrapperSingle(mybox->d_cpxGabe, mybox->d_cpxAlex, -1);

    //     d_cpxToFloatVecComponent<<<mybox->M_Grid, mybox->M_Block>>>(d_fA, mybox->d_cpxAlex, j, Dim, M);
    //     check_cudaError("placing component");
    // }



    // cudaMemcpy(fA, d_fA, M*Dim*sizeof(float), cudaMemcpyDeviceToHost);

    // // debug i/o and quit
    // mybox->writeFieldFloat("gaussianPotential.dat", ur);
    // float *tp;
    // tp = (float*) calloc(M, sizeof(float));
    // for ( int j=0 ; j<Dim ; j++ ) {
    //     for ( int i=0 ; i<M; i++ ) {
    //         tp[i] = fA[i*Dim+j];
    //     }
    //     std::string nm = "gaussianForce" + std::to_string(j) + ".dat";

    //     mybox->writeFieldFloat(nm.c_str(), tp);
    // }

    // free(tp);
    // die("field written?");

}