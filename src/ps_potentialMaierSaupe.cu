// Copyright (c) 2025 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).
#include "ps_potentialMaierSaupe.h"
#include "PS_Box.h"

NBMaier::NBMaier() {}
NBMaier::~NBMaier() {}

NBMaier::NBMaier(std::istringstream& iss, PS_Box* box) : PS_Potential(iss, box) {
    
    iss >> grpI;
    iss >> grpJ;

    iss >> Ao;
    iss >> sig2;
    sig2 *= sig2;

}


void NBMaier::initializePotential() {
    std::cout << "Initializing Gaussian potential..." << std::endl;

    PS_Potential::initializePotential();




    std::complex<float> I(0.0, 1.0);
    float kv[3], k2;
    int Dim = mybox->returnDimension();
    int M = mybox->M;

    for ( int i=0 ; i<M ; i++ ) {
        k2 = mybox->get_kD(i, kv);
        uk[i] = Ao * exp(-k2 * sig2 / 2.0f) ;
        
        // In k-space, f(k) = -I * k * u(k)
        for (int j = 0; j < Dim; j++) {
            fk[i * Dim + j] = -I * kv[j] * uk[i];
        }
    }
}



void NBMaier::CalcForces() {

}


float NBMaier::CalcEnergy() {
    return 666.0;
}


float NBMaier::CalculateMaxEigenValue() {

    return 666.0;
}


float NBMaier::CalculateOrderParameter() { 

    return 666.0;
}