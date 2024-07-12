// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

//////////////////////////////////////////////////
// ps_particles.cu              Rob Riggleman   //
//                              July 6, 2024    //
// Class that contains particle-level info for  //
// particle-based simulations (e.g., TILD).     //
//////////////////////////////////////////////////

#include "ps_particles.h"

void die(const char*);

PS_Particle::PS_Particle() {
    nBonds = nAngles = 0;
}
PS_Particle::~PS_Particle() {}



// Sets the sizes of the vectors
void PS_Particle::setSizes(int dim, int maxBonds, int maxAngles) {
    x.resize(dim);
    v.resize(dim);
    f.resize(dim);

    bondedTo.resize(maxBonds);
    bondType.resize(maxBonds);

    if ( maxAngles > 0 ) die("Angles not set up in particle class!");
}