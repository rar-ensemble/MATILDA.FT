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
    float kv[3], *ur2;
    int Dim = mybox->returnDimension();
    int M = mybox->M;
    
    float r0[3], ri[3], dr[3];
    r0[0] = r0[1] = r0[2] = 0.0f;

    ur2 = (float*) malloc( M * sizeof(float) );

    for ( int i=0 ; i<M ; i++ ) {
        
        mybox->get_rf(i, ri);

        float mdr2 = mybox->pbc_dr2(dr, ri, r0);
        float mdr = sqrtf(mdr2);

        // Multiplication by V ensures proper normalization when used in
        // Fourier space.
        ur[i]  = Ao * mybox->V * (1.0 - erf((mdr - Rp1)/(xi1)));
        ur2[i] = Ao * mybox->V * (1.0 - erf((mdr - Rp2)/(xi2)));

    }

    // d_ur = ur, copy real-space potential to device
    cudaMemcpy(d_ur, ur, M*sizeof(float), cudaMemcpyHostToDevice);

    // d_alex = d_ur
    d_floatToCpx<<<mybox->M_Grid, mybox->M_Block>>>(mybox->d_cpxAlex, d_ur, M);

    // gabe = FT(alex)
    mybox->cufftWrapperSingle(mybox->d_cpxAlex, mybox->d_cpxGabe, 1);



    // d_ur = ur2, copy real-space potential to device
    cudaMemcpy(d_ur, ur2, M*sizeof(float), cudaMemcpyHostToDevice);

    // d_alex = d_ur
    d_floatToCpx<<<mybox->M_Grid, mybox->M_Block>>>(mybox->d_cpxAlex, d_ur, M);

    // alex = FT(alex)
    mybox->cufftWrapperSingle(mybox->d_cpxAlex, mybox->d_cpxAlex, 1);


    // FT(d_uk) = FT(erf1)*FT(erf2), convolution of erf1 with erf2
    d_multiplyCpxByCpx<<<mybox->M_Grid, mybox->M_Block>>>(d_uk, mybox->d_cpxGabe, mybox->d_cpxAlex, M);



    // gabe = IFT(d_uk)
    mybox->cufftWrapperSingle(d_uk, mybox->d_cpxGabe, -1);

    // d_ur = real(gabe)
    d_cpxToFloat<<<mybox->M_Grid, mybox->M_Block>>>(d_ur, mybox->d_cpxGabe, M);
    // ur = d_ur
    cudaMemcpy(ur, d_ur, M*sizeof(float), cudaMemcpyDeviceToHost);


    // uk = d_uk
    cudaMemcpy(uk, d_uk, M*sizeof(std::complex<float>), cudaMemcpyDeviceToHost);


    // Define the forces in Fourier space
    for ( int i=0 ; i<M ; i++ ) {
        mybox->get_kD(i, kv);
        for (int j = 0; j < Dim; j++) {
            fk[i * Dim + j] = -I * kv[j] * uk[i];
        }
    }


    // Send force arrays to device
    // d_fk = fk
    cudaMemcpy(d_fk, fk, M*Dim*sizeof(std::complex<float>), cudaMemcpyHostToDevice);
    check_cudaError("sending force array to device in Gauss potential");

    free(ur2);

    std::string potName = "erf2-potential-" + grpI + "-" + grpJ + ".dat";
    mybox->writeFieldFloat(potName.c_str(), ur);
    //die("field written1");

}//initializePotential()