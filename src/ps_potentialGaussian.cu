// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "ps_potentialGaussian.h"

NBGauss::NBGauss() {}
NBGauss::~NBGauss() {}


// Constructor called by the "factor" routine in ps_potential.cu
NBGauss::NBGauss(std::istringstream& iss, PS_Box* box) : PS_Potential(iss, box) {
    std::cout << "Making Gaussian potential..." << std::endl;

    iss >> grpI;
    iss >> grpJ;

    iss >> Ao;

    float sigma;
    iss >> sigma;
    sig2 = sigma*sigma;
}

