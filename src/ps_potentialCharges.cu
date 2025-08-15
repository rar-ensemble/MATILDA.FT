// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "ps_potentialCharges.h"
#include "PS_Box.h"



NBCharge::NBCharge() {}
NBCharge::~NBCharge() {}

// Constructor called by the "factor" routine in ps_potential.cu
NBCharge::NBCharge(std::istringstream& iss, PS_Box* box) : PS_Potential(iss, box) {
    
    grpI = "charges";

    iss >> LB;

    float sigma;
    iss >> sigma;
    sig2 = sigma*sigma;
}


// uk will store the Green's function for Poisson's function
// real-space functions will not be defined
void NBCharge::initializePotential() {
    std::cout << "Initializing charge potential..." << std::endl;

    Iind = mybox->findGroupInteger(grpI);
    
    std::complex<float> I(0.0, 1.0);
    float kv[3], k2;
    int Dim = mybox->returnDimension();
    int M = mybox->M;

    for ( int i=0 ; i<M ; i++ ) {
        k2 = mybox->get_kD(i, kv);

        // The factor of exp(-k2 * sig2) accounts for smearing, including the smearing
        // required when mapping back to the particles for electric field
        if ( k2 > 0.0 ) 
            uk[i] = PI4 * LB / k2 * exp(-k2 * sig2 ) ;
        else 
            uk[i] = 0.0f;
        

        // In k-space, f(k) = -I * k * u(k)
        // The additional Gaussian is from the charge density of 
        // particle the force is mapped back onto
        for (int j = 0; j < Dim; j++) {
            fk[i * Dim + j] = -I * kv[j] * uk[i] ;
        }
    }

    // Send these to device, inv transform to get ur, f(r)
    cudaMemcpy(d_uk, uk, M*sizeof(std::complex<float>), cudaMemcpyHostToDevice);
    check_cudaError("uk --> d_uk in initialize Gaussian");


    // Same thing for force arrays
    cudaMemcpy(d_fk, fk, M*Dim*sizeof(std::complex<float>), cudaMemcpyHostToDevice);
    check_cudaError("sending force array to device in Gauss potential");


}//initializePotential()


float NBCharge::CalcEnergy() {
    float *d_rhoq;

    // Pointers to reused box vars
    cuComplex *d_cpxAlex, *d_cpxGabe;
    float *d_Gabe, *d_Alex;
    d_cpxAlex = mybox->d_cpxAlex;
    d_cpxGabe = mybox->d_cpxGabe;
    d_Gabe = mybox->d_Gabe;
    d_Alex = mybox->d_Alex;

    int M = mybox->M;
    int Dim = mybox->returnDimension();
    int Grid = mybox->M_Grid;
    int Block = mybox->M_Block;


    // Pointer to charge density field 
    d_rhoq = mybox->psGroup[Iind].d_rhoq;

    // real(Alex) = rhoJ, imag(Alex) = 0.0
    d_floatToCpx<<<Grid, Block>>>(d_cpxAlex, d_rhoq, M);

    // Gabe = FT(Alex=rhoJ); Alex now available
    mybox->cufftWrapperSingle(d_cpxAlex, d_cpxGabe, 1);

    check_cudaError("Charge potential first fft");


    // Alex = uk[j] * FT(rhoJ), j \in [x,y,z], Alex = FT(e potential)
    d_multiplyCpxByCpx<<<Grid, Block>>>(d_cpxAlex, d_uk, d_cpxGabe, M);
    
    
    // Gabe = IFT(Alex) = electrostatic potential
    mybox->cufftWrapperSingle(d_cpxAlex, d_cpxAlex, -1);
    d_cpxToFloat<<<Grid, Block>>>(d_Gabe, d_cpxAlex, M);




    // Alex = rhoq * Gabe
    d_multiplyFloatByFloat<<<Grid, Block>>>(d_Alex, d_rhoq, d_Gabe, M);

    // integrate the field, divide by 2
    energy = 0.5 * mybox->gvol * mybox->sumDeviceArray(d_Alex, mybox->M_Block, M);

 
    return this->energy;
}





void NBCharge::CalcForces() {
    float *d_rhoq;

    // Pointers to reused box vars
    cuComplex *d_cpxAlex, *d_cpxGabe;
    float *d_Gabe;
    d_cpxAlex = mybox->d_cpxAlex;
    d_cpxGabe = mybox->d_cpxGabe;
    d_Gabe = mybox->d_Gabe;

    int M = mybox->M;
    int Dim = mybox->returnDimension();
    int Grid = mybox->M_Grid;
    int Block = mybox->M_Block;


    // Pointer to density field for J
    d_rhoq = mybox->psGroup[Iind].d_rhoq;

    // real(Alex) = rhoJ, imag(Alex) = 0.0
    d_floatToCpx<<<Grid, Block>>>(d_cpxAlex, d_rhoq, M);

    // Gabe = FT(Alex=rhoJ); Alex now available
    mybox->cufftWrapperSingle(d_cpxAlex, d_cpxGabe, 1);

    check_cudaError("Charge potential first fft");

    for ( int j=0 ; j<Dim ; j++ ) {
        // Alex = fk[j] * FT(rhoq), j \in [x,y,z]
        d_multiplyCpxDirByCpx<<<Grid, Block>>>(d_cpxAlex, d_fk, d_cpxGabe, j, Dim, M);


        // Gabe = IFT(Alex)
        mybox->cufftWrapperSingle(d_cpxAlex, d_cpxAlex, -1);
        d_cpxToFloat<<<Grid, Block>>>(d_Gabe, d_cpxAlex, M);


        // Gabe now contains the forces that act on particles I
        mybox->psGroup[Iind].accumulateGridForceComp(d_Gabe, j);        
    }
}