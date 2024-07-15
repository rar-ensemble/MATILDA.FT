// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#ifndef _PS_BOX
#define _PS_BOX

#include "Box.h"
#include <cufft.h>
#include <cufftXt.h>

#include "ps_particles.h"
#include "ps_species.h"

class PS_Box : public Box {
    protected:
        std::string ftsStyle;              // "scft" or "cl", maybe also "hpf" in future?
    public:
        ~PS_Box();
        PS_Box(std::istringstream&);

        double rho0;        // System density
        double Nr;          // Reference chain length 
                            
        double C;           // System concentration
        double Utot;        // Total potential energy

        int nstot;           // Total number of particles/atoms
        int nBondsTot;      // Total number of bonds
        int nAnglesTot;      // Total number of bonds
        int logFreq;        // Frequency to write to ps_data.dat
        int gsdFreq;        // Frequency to write to gsd files
        int fieldFreq;      // Frequency to write density field data
        

        std::vector<PS_Particle> partic;    // vector of particle info
        std::vector<PS_Species> species;    // vector of species IDs
                

        void makeLinear(std::istringstream&);   // Create linear multiblock copolymer
        
        int findSpeciesInteger(std::string);

        std::ofstream OTP;
        void readInput(std::ifstream&);
        void doTimeStep(int);
        void initializeSim() override;
        void writeData(int) override;
        void writeFields() override;
        void writeTime() override;
        int converged(int dm) {return 0; };
        void writeDataConfig(std::string);

};

#endif // FTS_BOX
