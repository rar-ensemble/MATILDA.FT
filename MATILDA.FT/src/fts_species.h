// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).



////////////////////////////////////////////////
// fts_species.h               R. Riggleman    //
// 9 Aug 2022                                  //
// Class to define species Species and contain //
// their density fields. For example, if one   //
// simulates at A + A/B blend, that box        //
// would have two FTS_Species (A and B), and   //
// the total density field for A would be      //
// contained herein. A homopolymer density     //
// would be contained in the homopolymer       //
// FTS_Molecule.                               //
/////////////////////////////////////////////////

#ifndef FTS_TYPE
#define FTS_TYPE
#include "include_libs.h"
#include "fts_potential.h"
#include "FTS_Box.h"

#include <istream>


class FTS_Species {
    protected:
        std::string input_command;
        FTS_Box *box;
    public:
        std::string fts_species;

        // density, d_density is the total density field associated with this spcies
        // If shape functions are used, they should be the total density, not the
        // center density.
        thrust::host_vector<thrust::complex<double>> density;
        thrust::device_vector<thrust::complex<double>> d_density;
        
        // d_Ak is the total linear coefficient for the semi-implicit method
        // for the A component. Each molecule should add its contribution to d_Ak;
        // This is to be stored in k-space.
        thrust::device_vector<thrust::complex<double>> d_Ak;

        // "Bare" field for this species (i.e., not convolved with a shape function)
        thrust::device_vector<thrust::complex<double>> d_w;

        FTS_Species();
        FTS_Species(std::istringstream&, FTS_Box*);
        virtual ~FTS_Species();     
        void buildPotentialField(); // Loops over the potentials and constructs the field for each species
        void zeroDensity();         // Zeroes the species density fields
        void writeDensity(const int);
        void writeSpeciesFields(const int);
        std::string printCommand() {return input_command;}

};


#endif