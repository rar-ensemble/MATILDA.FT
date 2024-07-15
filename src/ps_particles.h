// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

//////////////////////////////////////////////////
// ps_particles.h               Rob Riggleman   //
//                              July 6, 2024    //
// Class that contains particle-level info for  //
// particle-based simulations (e.g., TILD).     //
//////////////////////////////////////////////////

#ifndef _PS_PARTICS
#define _PS_PARTICS

#include "include_libs.h"

class PS_Box;

class PS_Particle {
    public:
        std::string species;   // Particle type as a string
        int intSpecies;        // Particle type index
        int mID;               // Molecule ID

        std::vector<float> x;   // [Dim] Store particle position
        std::vector<float> v;   // [Dim] store particle velo
        std::vector<float> f;   // [Dim] store particle forces

        int nBonds;                 // number of bonds for this particle
        std::vector<int> bondedTo;  // bond partners for this particle
        std::vector<int> bondType;  // bond type for this pair

        int nAngles;                // number of angles for this particle

        PS_Box* mybox;
        PS_Particle();
        virtual ~PS_Particle();
        void setSizes(int,int,int);

};

#endif