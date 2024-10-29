// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#ifndef _PS_BOX
#define _PS_BOX

#include "Box.h"
#include <cufft.h>
#include <cufftXt.h>

#include "ps_species.h"
#include "ps_groups.h"
#include "ps_integrator.h"
#include "ps_potential.h"

class PS_Box : public Box {
    protected:
    public:
        ~PS_Box();
        PS_Box(std::istringstream&);

        double rho0;        // System density
        double Nr;          // Reference chain length 
                            
        double C;           // System concentration
        double Utot;        // Total potential energy

        int nstot;          // Total number of particles/atoms
        int nBondsTot;      // Total number of bonds
        int nBondTypes;     // number of bond types
        int nAnglesTot;     // Total number of angles
        int nAngleTypes;    // number of angle types
        int nTypes;         // Total number of particle types
        int logFreq;        // Frequency to write to ps_data.dat
        int gsdFreq;        // Frequency to write to gsd files
        int trajFreq;       // Frequency to write to lammpstrj file
        int fieldFreq;      // Frequency to write density field data
        int pmeorder;       // Order of the PME interpolation
        int gridPerPartic;  // Number of grid points each particle interacts
        int MAXBONDS;       // Max number of bonds per particle
        int MAXANGLES;      // Max number of angles per particle
        int NSEXTRA;        // Extra memory allocated to store particles
        int nsGrid;         // GPU grid number for 'all' particle operations
        int DnsGrid;        // GPU grid number for Dim*all particle ops
        int nsBlock;        // GPU block number for 'all' particle operations
        int RANDSEED;       // Seed for CUDA RNG
        int doCharges;      // Flag for whether charge species exist or not
        int n_P_comps;      // Number of independent pressure components (3 or 6)
        int nMolecules;     // Number of molecules in the box, not sure needed/used
        
        curandState* d_states; // [Dim*nstot] State var. for particle-level RNG
        
        float Ubond, Uangle;// Stores bond and angle energy

        // Data for gsd, lammpstrj file storage
        std::string gsd_name, trajFileName;

        std::vector<unsigned int> list_of_bond_type;        // [nBondsTot] bond storage for gsd file
        std::vector<unsigned int> list_of_bond_partners;    // [nBondsTot*2] bond storage for gsd file
        
        std::vector<unsigned int> list_of_angle_type;       // [nAnglesTot] angle storage for gsd file
        std::vector<unsigned int> list_of_angle_partners;   // [nAnglesTot*3] angle storage for gsd file

        float* d_dxf;      // float version of grid spacing
        
        thrust::host_vector<float> x;       // [nstot*Dim] particle positions 
        float *d_x;                         // [nstot*Dim] device particle positions  

        float *f, *d_f;                     // [nstot*Dim]
        float *v, *d_v;                     // [nstot*Dim]
        // thrust::host_vector<float> f;       // [nstot*Dim] particle forces 
        // thrust::device_vector<float> d_f;   // [nstot*Dim] device particle forces
        // float* _d_f;           

        std::vector<PS_Species> species;            // vector of species IDs
        float *speciesMass, *d_speciesMass;          // [nTypes] masses of species
        float *speciesMobility, *d_speciesMobility;  // [nTypes] mobility (diffusivity) of species
        
        int *nBonds, *d_nBonds;         // [nstot] device number of bonds 
        int *bondedTo, *d_bondedTo;     // [MAXBONDS*nstot] device bond partner list
        int *bondType, *d_bondType;     // [MAXBONDS*nstot] device bond types for particles
        int bondTime;                   // Stores time spent in bond function


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


                      // pointer to d_nBonds.data()

        thrust::host_vector<float> bondReq;     // [nBondTypes] equil dist for bonds
        float* d_bondReq;                      // [nBondTypes] equil dist for bonds

        thrust::host_vector<float> bondK;       // [nBondTypes] force const for bonds
        float* d_bondK;                        // [nBondTypes] force const for bonds

        thrust::host_vector<int> bondStyle;     // [nBondTypes] bond type (1=harmonic, 2=FENE)  
        int* d_bondStyle;



        thrust::host_vector<int> nAngles;        // [nstot] number of bonds per particle vector
        thrust::device_vector<int> d_nAngles;    // [nstot] number of bonds per particle vector

        thrust::host_vector<int> angleGroup;    // [nstot*MAXANGLES*3] list of the three particles in each angle
        thrust::device_vector<int> d_angleGroup;// [nstot*MAXANGLES*3] list of the three particles in each angle

        thrust::host_vector<int> angleType;     // [MAXANGLES*nstot] bond types for each particles
        thrust::device_vector<int> d_angleType; // [MAXANGLES*nstot] device, bond types for particles
        
        thrust::host_vector<float> angleTheq;     // [nAngleTypes] equil angle
        thrust::device_vector<float> d_angleTheq; // [nAngleTypes] 
        
        thrust::host_vector<float> angleK;       // [nAngleTypes] force const for angles
        thrust::device_vector<float> d_angleK;   // [nAngleTypes] force const for angles
        
        thrust::host_vector<int> angleStyle;     // [nAngleTypes] angle type (0=WLC, 1=harmonic)  
        thrust::device_vector<int> d_angleStyle; // [nAngleTypes] angle type (0=WLC, 1=harmonic)  


        // Variables named for G&A


        std::vector<PS_Group> psGroup;          // Vector of particle groups
        
        std::vector<Integrator*> integrators;   // Time integration schemes
        std::vector<PS_Potential*> potentials;   // 


        void allocHostParticleArrays(int);      // Uses 'resize' to allocate host particle arrays
        void allocDeviceParticleArrays(int);    // Uses 'resize' to allocate device particle arrays
        void sendAllHostToDevice(void);         // Sends all particle-sized arrays from host to dev





        void GSDinit(void);
        void writeGSDtraj(void);
        void writeLammpsTraj(int);
        void readGSDtraj(const char*, int, int);
             


        int findSpeciesInteger(std::string);
        int findGroupInteger(std::string);

        void NVT(int) override;
        void forces(void);
        void computeThermoProps(void);

        std::ofstream OTP;
        void readInput(std::ifstream&);     // Reads the input file
        void doTimeStep(int);               // Performs one time step of a sim
        void initializeSim() override;      // Initializes files prior to beginning simulation
        void writeData(int) override;       

        void writeFields() override;        // Loops over groups, writes to data file
        void writeTime() override;          // Writes subroutine run times
        int converged(int dm) {return 0; }; // No implemented for PS methods
        void writeDataConfig(std::string);  // Writes LAMMPS data file format
        void createDefaultGroups();         // Makes default groups

        void finishInitialization();          // Finishes initializing box after reading input

        // Writes thrust::host_vector array
        void writeFieldTFloat(const char*, thrust::host_vector<float>);        
        void writeFieldFloat(const char*, const float*);

        void makeLinear(std::istringstream&);   // Create linear multiblock copolymer

};

#endif // FTS_BOX
