// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "ps_potentialErf2.h"
#include "PS_Box.h"



NBErf2::NBErf2() {}
NBErf2::~NBErf2() {}

// Constructor called by the "factor" routine in ps_potential.cu
NBErf2::NBErf2(std::istringstream& iss, PS_Box* box) : PS_Potential(iss, box) {
    
    iss >> grpI;
    iss >> grpJ;

    iss >> Ao;
    iss >> Rp1;
    iss >> xi1;
    iss >> Rp2;
    iss >> xi2;
}

void NBErf2::initializePotential() {
    std::cout << "Initializing Erf potential..." << std::endl;

    PS_Potential::initializePotential();
    
    std::complex<float> I(0.0, 1.0);
    float kv[3], k2, kmag;
    int Dim = mybox->returnDimension();
    int M = mybox->M;
    
    for ( int i=0 ; i<M ; i++ ) {
        k2 = mybox->get_kD(i, kv);
        kmag = sqrt(k2);

        if (k2 == 0) {
            uk[i] = Ao *				// prefactor
            PI4 * Rp1 * Rp1 * Rp1 / 3*   // step function contribution for 1
            PI4 * Rp2 * Rp2 * Rp2 / 3;   // step function contribution for 2
        }
        else
        {
            //FFT of step function 
            float temp1 = PI4 * (sin(Rp1 * kmag) - Rp1 * kmag * cos(Rp1 * kmag)) / (k2 * kmag);
            float temp2 = PI4 * (sin(Rp2 * kmag) - Rp2 * kmag * cos(Rp2 * kmag)) / (k2 * kmag);

            uk[i] = Ao *				//prefactor
                exp(-k2 * (xi1*xi1 + xi2*xi2) * 0.5f) * //Gaussian contribution of both
                temp1 * // step function for 1
                temp2 ; // step function for the other
        }
        
        for (int j = 0; j < Dim; j++) {
            fk[i * Dim + j] = -I * kv[j] * uk[i];
        }

    }// k=0:M


    // Send these to device, inv transform to get ur, f(r)
    cudaMemcpy(d_uk, uk, M*sizeof(std::complex<float>), cudaMemcpyHostToDevice);
    check_cudaError("uk --> d_uk in initialize Gaussian");

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
    std::cout << "  erf2 initialization completed" << std::endl;


    std::complex<float> *tp;
    tp = (std::complex<float>*) malloc(M*sizeof(std::complex<float>));

    for ( int j=0 ; j<Dim ; j++ ) {
        
        for ( int i=0 ; i<M ; i++ ) {
            tp[i] = fk[i*Dim+j];
        }
        
        cudaMemcpy(mybox->d_cpxAlex, tp, M*sizeof(std::complex<float>), cudaMemcpyHostToDevice);
        check_cudaError("erf2 test");
        
        mybox->cufftWrapperSingle(mybox->d_cpxAlex, mybox->d_cpxGabe, 1);
        check_cudaError("erf2 test 1");

        d_cpxToFloat<<<mybox->M_Grid, mybox->M_Block>>>(mybox->d_Alex, mybox->d_cpxGabe, M);
        check_cudaError("erf2 test 2");

        cudaMemcpy(mybox->alex, mybox->d_Alex, M*sizeof(float), cudaMemcpyDeviceToHost);
        check_cudaError("erf2 test 3");

        std::string tName = "erf2-force-" + std::to_string(j) + ".dat";
        mybox->writeFieldFloat(tName.c_str(), mybox->alex);


    }

    std::string potName = "erf2-potential-" + grpI + "-" + grpJ + ".dat";
    mybox->writeFieldFloat(potName.c_str(), ur);
    free(tp);


}//initializePotential()