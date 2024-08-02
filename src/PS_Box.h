// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#ifndef _PS_BOX
#define _PS_BOX

#include "Box.h"
#include <cufft.h>
#include <cufftXt.h>

#include "ps_species.h"
#include "ps_groups.h"

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

        int nstot;          // Total number of particles/atoms
        int nBondsTot;      // Total number of bonds
        int nAnglesTot;     // Total number of angles
        int nTypes;         // Total number of particle types
        int logFreq;        // Frequency to write to ps_data.dat
        int gsdFreq;        // Frequency to write to gsd files
        int fieldFreq;      // Frequency to write density field data
        int pmeorder;       // Order of the PME interpolation
        int gridPerPartic;  // Number of grid points each particle interacts
        int MAXBONDS;       // Max number of bonds per particle
        int MAXANGLES;      // Max number of angles per particle
        int nsGrid;         // GPU grid number for 'all' particle operations
        int DnsGrid;        // GPU grid number for Dim*all particle ops
        int nsBlock;        // GPU block number for 'all' particle operations
        curandState* d_states; // State var. for particle-level RNG
        int RANDSEED;       // Seed for CUDA RNG

        float* _d_dxf;      // float version of grid spacing
        
        thrust::host_vector<float> x;       // [nstot*Dim] particle positions 
        thrust::device_vector<float> d_x;   // [nstot*Dim] device particle positions
        float* _d_x;                        // Pointer to d_x.data() 
        
        thrust::host_vector<float> v;       // [nstot*Dim] particle velocities
        thrust::device_vector<float> d_v;   // [nstot*Dim] devoce particle velocities
        float* _d_v;                        // Pointer to d_v.data() 
        
        thrust::host_vector<float> f;       // [nstot*Dim] particle forces 
        thrust::device_vector<float> d_f;   // [nstot*Dim] device particle forces
        float* _d_f;                        // Pointer to d_f.data() 

        thrust::host_vector<int> intSpecies;        // [nstot] particle type index
        thrust::device_vector<int> d_intSpecies;    // [nstot] device particle type index
        int* _d_intSpecies;                         // pointer to d_intSpecies.data()

        thrust::host_vector<int> mID;               // [nstot] molecule index
        thrust::device_vector<int> d_mID;           // [nstot] device molecule index
        int* _d_mID;                                // pointer to d_mID.data()

        thrust::device_vector<float> d_gridW;   // [nstot*gridPerPartic] particle-to-grid weights
        float* _d_gridW;                        // Pointer to d_gridW.data()

        thrust::device_vector<int> d_gridInds;  // [nstot*gridPerPartic] particle-to-grid indices
        int* _d_gridInds;                        // Pointer to d_gridInds.data()

        thrust::host_vector<int> nBonds;        // [nstot] number of bonds 
        thrust::device_vector<int> d_nBonds;    // [nstot] device number of bonds 
        int* _d_nBonds;                         // pointer to d_nBonds.data()

        thrust::host_vector<int> bondedTo;    // [MAXBONDS*nstot] bond partner list
        thrust::device_vector<int> d_bondedTo;// [MAXBONDS*nstot] device bond partner list
        int* _d_bondedTo;                         // pointer to d_nBonds.data()

        thrust::host_vector<int> bondType;    // [MAXBONDS*nstot] bond types for each particles
        thrust::device_vector<int> d_bondType;// [MAXBONDS*nstot] device, bond types for particles
        int* _d_bondType;                         // pointer to d_nBonds.data()

        thrust::host_vector<int> nAngles;        // [nstot] number of bonds per particle vector
        thrust::device_vector<int> d_nAngles;    // [nstot] number of bonds per particle vector


        // Variables named for G&A

        std::vector<PS_Species> species;    // vector of species IDs
        thrust::host_vector<float> speciesMass;
        thrust::device_vector<float> d_speciesMass;
        float* _d_speciesMass;

        thrust::host_vector<float> speciesMobility;
        thrust::device_vector<float> d_speciesMobility;
        float* _d_speciesMobility;

        std::vector<PS_Group> psGroup;        // Vector of particle groups
                
        void allocHostParticleArrays(int);      // Uses 'resize' to allocate host particle arrays
        void allocDeviceParticleArrays(int);    // Uses 'resize' to allocate device particle arrays
        void sendAllHostToDevice(void);         // Sends all particle-sized arrays from host to dev

             
        int findSpeciesInteger(std::string);
        int findGroupInteger(std::string);

        void NVT(int) override;
        std::ofstream OTP;
        void readInput(std::ifstream&);     // Reads the input file
        void doTimeStep(int);               // Performs one time step of a sim
        void initializeSim() override;      // Initializes files prior to beginning simulation
        void writeData(int) override;
        void writeFields() override;        // Currently does nothing?
        void writeTime() override;          // Writes subroutine run times
        int converged(int dm) {return 0; }; // No implemented for PS methods
        void writeDataConfig(std::string);  // Writes LAMMPS data file format
        void createDefaultGroups();         // Makes default groups
        void finishSpeciesArrays();          // Allocates and populates any needed static arrays


        void makeLinear(std::istringstream&);   // Create linear multiblock copolymer

};

#endif // FTS_BOX
