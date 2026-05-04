// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "ps_potentialLangevin.h"
#include "PS_Box.h"

__global__ void d_add_Langevin_forces(float*, const float*, const int*, const float,const float, 
    const int, const int,curandState* );

/*
This `potential' adds Langevin forces to the particles. This allows use of 
the velocity Verlet algorithm for integration and primarily can be used
to compare convergence against other algorithms (DPD) and integrators (GJF).
HOW DO I AUTOMATICALLY PASS THE TIME STEP SIZE TO THIS ROUTINE??
*/
Langevin::Langevin(std::istringstream& iss, PS_Box* box) : PS_Potential(iss, box) {

    // Record the species acted upon
    iss >> grpI ;
    grpJ = grpI;


    // Set default friction coefficient magnitude
    drag = 1.0;

    // delt initialized to mybox->integrator[0].delt;
    if ( mybox->integrators.size() > 0 ) {
        delt = mybox->integrators[0]->delt;
    }
    else {
        delt = -5.0;
    }

    
    // parse optional arguments
    while ( iss.tellg() != -1 ) {
        std::string temp_str;

        iss >> temp_str;
        if ( temp_str == "drag" ) {
            iss >> drag;
        }

        else if ( temp_str == "delt" ) {
            iss >> delt;
        }

        else {
            std::string err_msg = temp_str + " is not a valid initialize option in fts_potential";
            die(err_msg.c_str());
        }
    } // while (!iss)

    if ( delt < 0.0 ) {
        std::string last_words = "ps-potentialLangevin.cu: did not find appropriate delt for the Langevin noise";
        die(last_words);
    }

}


void Langevin::initializePotential() {

    PS_Potential::initializePotential();

}


// Accumulate the forces from the bias field on group Iind
void Langevin::CalcForces() {

    noise_mag = sqrtf(2.0 *drag / delt );

    int GRID = mybox->psGroup[Iind].Grid;
    int BLOCK = mybox->psGroup[Iind].Block;
    int Dim = mybox->returnDimension();
    int ns = mybox->psGroup[Iind].nsites;

    d_add_Langevin_forces<<<GRID, BLOCK>>>(mybox->d_f, mybox->d_v, mybox->psGroup[Iind].d_siteList, drag, noise_mag,
        Dim, ns, mybox->d_states);
}


// Returns 0 energy because these forces do not arise from a conservative potential
float Langevin::CalcEnergy() {
    return 0.0f;
}

Langevin::Langevin() {

}

Langevin::~Langevin() {

}




__global__ void d_add_Langevin_forces(
    float* d_f,             // [Dim*nstot] particle forces
    const float* d_v,       // [Dim*nstot] particle velocities
    const int* sites,       // [ns] indices of particles in the group
    const float drag,       // Friction coefficient
    const float noise_mag,  // noise magnitude
    const int Dim,          // system dimension
    const int ns,           // number of particles
    curandState* d_states   // [nstot] RNG state vector
) {

    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= ns)
        return;

    // Particle index 
    int pind = sites[id];

    curandState l_state = d_states[pind];

    // 0:Dim*nstot index
    int ind = pind * Dim;
    for ( int j=0 ; j<Dim ; j++ ) {

        d_f[ind+j] += -d_v[ind+j] * drag + noise_mag * curand_normal(&l_state);
    }

    d_states[pind] = l_state;

}