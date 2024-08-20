// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "PS_Box.h"
#include "random.h"
#include "include_libs.h"
#include "gsd.h"
#include <algorithm>
#include <map>

void die(const char*);
double ran2(void);
void random_unit_vec(double*, int);

__global__ void d_calcGridWeights(float*, int*, const float*, const int*, 
const float*, const int, const int, const int, const int);
__global__ void d_bonds(float*, const int*, const int*, const int*,
const float*, const float*, const int*, const float*, const float*,
const float*, const int, const int, const int);
__global__ void d_initDeviceRNG(unsigned int, curandState*, int);

Integrator* IntegratorFactory(std::istringstream&, PS_Box*);

// Executes the commands for a given time step
// Updates all fields, then recomputes all molecule densities
// then populating species densities
void PS_Box::doTimeStep(int step) {


    // First integration step, when needed (e.g., velo Verlet)
    for ( int i=0 ; i<integrators.size(); i++ ) {
        integrators[i]->Integrate_1();
    }


    // Update grid weights
    d_calcGridWeights<<<nsGrid, nsBlock>>>(_d_gridW, _d_gridInds, _d_x, _d_Nx, 
        _d_dxf, nstot, pmeorder, M, Dim );

    // update the density fields
    for ( int i=0 ; i<psGroup.size(); i++ ) {
        // zero density, grid force fields
        psGroup[i].zeroFields();

        // Fill density field
        psGroup[i].makeDensityField();
    }

    // Zero particle forces
    d_assignFloatVal<<<DnsGrid, nsBlock>>>(_d_f, 0.0, Dim*nstot);

    forces();


    // Second integration step
    for ( int i=0 ; i<integrators.size(); i++ ) {
        integrators[i]->Integrate_2();
    }


    // Write log data
    if ( step % logFreq == 0 ) {
        writeData(step);
    }

} // doTimeStep


void PS_Box::NVT(int maxSteps) {
    std::cout << "RUNNING NVT?!?" << std::endl;
    
    for ( int i=0 ; i<maxSteps; i++ ) {
        doTimeStep(i);

        totSteps++;
    }
}


void PS_Box::forces() {
    
    // 3. bonded forces; 

    // 4. NB forces; 
    // 5. Extras

}


// Write Hamiltonian terms to output file
void PS_Box::writeData(int step) {

    // computeHamiltonian();

    OTP.open("ps_data.dat", std::ios_base::app);
    std::string outline;

    OTP << step ;

    OTP << std::endl;
    std::cout << std::endl;

    OTP.close();
}


void PS_Box::initializeSim() {
    

    // Initialize the output stream
    OTP.open("ps_data.dat");
    OTP.close();

    totSteps = 0;
}


// Reads all of the commands from the input file from when this
// box is created (using the 'box' command) until the 'endBox' 
// command is caught. 
// This routine also serves to initialize the simulation box
// in preparation for a run.
void PS_Box::readInput(std::ifstream& inp) {
    std::cout << "In read input!" << std::endl;

    // Set some preliminary/default variables
    M = 1;
    V = 1.0;
    blockSize = 512;
    MAXBONDS = 2;
    MAXANGLES = 3;
    Nr = 100;
    rho0 = C = -1.0;
    nstot = nBondsTot = nAnglesTot = 0;

    // Some default values
    logFreq = 100;
    gsdFreq = 0;
    fieldFreq = 0;
    gsd_name = "traj.gsd";
    doCharges = 0;

    std::string line, firstWord;

    bool readDimension = false;

    while (!inp.eof()) {
        getline(inp, line);

        if ( line.length() == 0 || line.at(0) == '#')
            continue;

        std::istringstream iss(line);
        
        while ( iss >> firstWord ) {
            // std::cout << "Line read: " << line << std::endl;

            if ( firstWord == "endBox" ) {
                break;
            }

            if ( firstWord == "blocksize" || firstWord == "blockSize" ) { iss >> blockSize ; }


            else if ( firstWord == "bond" ) {
                int btype;
                iss >> btype;
                if ( btype > bondK.size() ) {
                    bondK.resize(btype);
                    bondReq.resize(btype);
                    bondStyle.resize(btype);
                }

                std::string style;
                iss >> style;

                iss >> bondK[btype-1];
                iss >> bondReq[btype-1];

                if ( style == "harmonic" ) {
                    bondStyle[btype-1] = 0;
                }
                else if ( style == "fene" || style == "FENE" ) {
                    bondStyle[btype-1] = 1;
                }
            }

            // Commands are alphabetical from here on
            else if ( firstWord == "boxLengths" ) {
                if (!readDimension) 
                    std::cout << "\nWARNING! WARNING!\nboxLengths read before dimension specified, assuming default value of 2"<<std::endl;

                for ( int j=0 ; j<Dim ; j++ ) {
                    iss >> L[j];
                    V *= L[j];
                    if ( Nx[0] > 0 ) { dx[j] = L[j] / double(Nx[j]); }
                }
            }


            else if ( firstWord == "Dim" ) {
                iss >> Dim;
                setDimension(Dim);
                readDimension = true;
                if ( Dim == 2 ) n_P_comps = 3;
                else n_P_comps = 6;
            }

            else if ( firstWord == "grid" ) {
                if ( !readDimension ) { die("Dim must be defined before grid!" );}
                for ( int j=0 ; j<Dim ; j++ ) {
                    iss >> Nx[j];
                    M *= Nx[j];
                    if ( L[0] > 0.0 ) { dx[j] = L[j] / double(Nx[j]); }
                }
            }

            else if ( firstWord == "gsdFreq" || firstWord == "gsd_freq" ) {
                iss >> gsdFreq;
            }

            else if ( firstWord == "gsdName" || firstWord == "gsd_name" ) {
                iss >> gsd_name;
            }

            else if ( firstWord == "integrator" ) {
                integrators.push_back( IntegratorFactory(iss, this) );
            }

            else if ( firstWord == "logFreq" || firstWord == "logfreq" ) { 
                iss >> logFreq; 
            }

            else if ( firstWord == "MAXANGLES" ) {
                if ( nstot > 0 ) die("MAXANGLES must be set before defining any molecules");
                iss >> MAXANGLES;
            }

            else if ( firstWord == "MAXBONDS" ) {
                if ( nstot > 0 ) die("MAXBONDS must be set before defining any molecules");
                iss >> MAXBONDS;
            }

            else if ( firstWord == "molecule" ) {
                std::string nextWord;
                iss >> nextWord;
                if ( nextWord == "linear" ) {
                    makeLinear(iss);
                }
            }

            else if ( firstWord == "Nr" ) {
                iss >> Nr;
            }

            else if ( firstWord == "Nx") {
                iss >> Nx[0];
                M *= Nx[0];
            }

            else if (firstWord == "Ny") {
                iss >> Nx[1];
                M *= Nx[1];
            }

            else if (firstWord == "Nz") {
                if ( Dim != 3 ) die("Nz supplied in a non-3D simulation!");
                iss >> Nx[2];
                M *= Nx[2];
            }

            else if ( firstWord == "pmeorder" ) {
                iss >> pmeorder;
            }


            else if (firstWord == "randSeed" || firstWord == "RAND_SEED" || firstWord == "RANDSEED") {
                iss >> idum;        // Set CPU RNG to have seed = RANDSEED
                RANDSEED = idum;    // Set GPU RNG to have seed = RANDSEED
            }

            else if (firstWord == "rho0") {
                iss >> rho0;
            }

            else if ( firstWord == "species" ) {
                species.push_back(PS_Species(iss, this));
            }

            else {
                std::string s1 = "Invalid keyword " + firstWord + " in FTS_Box::readInput()";
                die(s1.c_str());
            }
            std::cout << "Finished input line: " << line << std::endl;

        }// while ( iss >> firstWord && firstWord != "endBox" ) 
        
        
        if ( firstWord == "endBox" ) {
            std::cout << "endBox caught" << std::endl;
            break;
        }

    }// while (!inp.eof()), finished reading up to 'endBox' or end of file


    finishInitialization();
    simTime = time(0);

}// End of readInput()





// After box is created by input file, this completes the initialization
void PS_Box::finishInitialization() {

    if ( nstot == 0 ) {
        die("Box created with no particles?!?");
    }

    // After input read, make the FFT plan
    // This currently assumes complex-double to complex-double transforms
    // change Z2Z to C2C to switch to cpx-float
    if ( this->Dim == 2 ) 
        cufftPlan2d(&fftplan, Nx[1], Nx[0], CUFFT_Z2Z);
    if ( this->Dim == 3 ) 
        cufftPlan3d(&fftplan, Nx[2], Nx[1], Nx[0], CUFFT_Z2Z);

    // Define gvol, dx, gridPerPartic
    gvol = 1.0;
    gridPerPartic = 1;
    for ( int j=0 ; j<Dim ; j++ ) {
        dx[j] = L[j] / double(Nx[j]);
        gvol *= dx[j];

        gridPerPartic *= (pmeorder+1);
    }

    // gpuGrid, block sizes
    M_Block = blockSize;
    M_Grid = (int)ceil((double)(M) / M_Block);

    nsBlock = blockSize;
    nsGrid = (int)ceil((double)(nstot) / nsBlock);
    DnsGrid = (int)ceil((double)(Dim*nstot) / nsBlock);


    // Define C, rho0 depending on what is given
    if ( rho0 > 0 && C > 0 ) { die("Cannot define both C and rho0!"); }

    double Rg = pow( (Nr-1.0)/6.0, 0.5);
    if ( rho0 > 0 ) {
        C = rho0/Nr;
        for ( int j=0; j<Dim ; j++ ) C *= Rg;
        std::cout << "Using Nr = " << Nr << ", computed C = " << C << std::endl;
    }
    else if ( C > 0 ) {
        rho0 = C * Rg * Rg * Rg / Nr;
        std::cout << "Using Nr = " << Nr << ", computed rho0 = " << rho0 << " [b^-3]" << std::endl;
    }


    // Count number of bonds, angles
    for ( int i=0 ; i<nstot ; i++ ) {
        for ( int j=0 ; j<nBonds[i] ; j++ ) {
            if ( bondedTo[i*MAXBONDS+j] > i ) nBondsTot++;
        }

        for ( int j=0 ; j<nAngles[i] ; j++ ) {
            die("angle counter not set up in PS_Box.cu::readInput");
        }
    }

    writeDataConfig("init.input.data");
    std::cout << "Initial config in data file format written to init.input.data" << std::endl;

    // Finish memory allocation on host
    allocHostParticleArrays(nstot);

    // Allocate device memory and copy device vars
    allocDeviceParticleArrays(nstot);
    
    createDefaultGroups();



    // Assign groups to integrators
    for ( int i=0 ; i<integrators.size(); i++ ) {
        integrators[i]->findGroup();
    }


    // Complete initialization of species variables
    nTypes = species.size();

    speciesMass.resize(nTypes);
    speciesMobility.resize(nTypes);
    for ( int i=0 ; i<nTypes; i++ ) {
        speciesMass[i] = species[i].mass;
        speciesMobility[i] = species[i].mobility;
    }

    d_speciesMass.resize(nTypes);
    d_speciesMobility.resize(nTypes);

    d_speciesMass = speciesMass;
    d_speciesMobility = speciesMobility;

    _d_speciesMass = thrust::raw_pointer_cast(d_speciesMass.data());
    _d_speciesMobility = thrust::raw_pointer_cast(d_speciesMobility.data());
}


// Creates groups for each particle type and 'all'
void PS_Box::createDefaultGroups() {
    psGroup.push_back(PS_Group("all", -1, this));

    for ( int i=0 ; i<species.size(); i++ ) {
        psGroup.push_back(PS_Group("type", i, this));

        std::cout << "Group name: " << psGroup[i+1].returnName() << std::endl;
    }
    std::cout << "Groups for all, each type created" << std::endl;
}


void PS_Box::writeFields() {
}

void PS_Box::writeTime() {

    int dt = time(0) - simTime;
    std::cout << "Total simulation time: " << dt / 60 << "m" << dt % 60 << "sec" << std::endl;
    
    dt = ftTimer;
    std::cout << "Total FT time: " << dt / 60 << "m" << dt % 60 << "sec" << std::endl;

}



PS_Box::~PS_Box() {}

PS_Box::PS_Box(std::istringstream& iss ) : Box(iss) {
    std::string s1;

    std::cout << "Made PS_Box " << std::endl;
}


// Finds the index in the species vector with the label 'testLabel'
int PS_Box::findSpeciesInteger(std::string testLabel) {
    int id = -1;
    for ( int i=0 ; i<species.size() ; i++ ) {
        if ( species[i].isSpecies( testLabel) ) {
            id = i;
            break;
        }
    }
    if ( id < 0 ) die("Species label not found!");

    return id;
}

// Finds the index in the group vector with the label 'testLabel'
int PS_Box::findGroupInteger(std::string testLabel) {
    int id = -1;
    for ( int i=0 ; i<psGroup.size() ; i++ ) {
        if ( psGroup[i].isGroup( testLabel) ) {
            id = i;
            break;
        }
    }
    if ( id < 0 ) die("Species label not found!");

    return id;
}

void PS_Box::writeDataConfig(std::string filename) {
    
    std::ofstream out(filename);

    out << "Created by MATILDA.FT\n\n" ;

    out << nstot << " atoms" << std::endl;
    out << nBondsTot << " bonds" << std::endl;
    out << nAnglesTot << " angles" << std::endl;

    // blank line
    out << std::endl;

    // Find number of particle species
    int max = -342332;
    for ( int i=0 ; i<nstot ; i++ ) {
        if ( intSpecies[i] > max ) max = intSpecies[i];
    }
    out << max+1 << " atom types" << std::endl;

    // Find number of bond types
    max = -1;
    for ( int i=0 ; i<nstot ; i++ ) {
        for ( int j=0 ; j<nBonds[i]; j++ ) 
            if ( bondType[i*MAXBONDS+j] > max ) max = bondType[i*MAXBONDS+j];
    }
    out << max+1 << " bond types" << std::endl;

    // Find number of angle types
    max = -1;
    for ( int i=0 ; i<nstot ; i++ ) {
        for ( int j=0 ; j<nAngles[i]; j++ ) 
            die("angles not set up in writeDataCofnig");
    }
    out << max+1 << " angle types" << std::endl;


    // blank line
    out << std::endl;

    // Box dimensions
    out << "0.0 " << L[0] << " xlo xhi" << std::endl;
    out << "0.0 " << L[1] << " ylo yhi" << std::endl;
    if ( Dim > 2 ) out << "0.0 " << L[2] << " zlo zhi" << std::endl;
    else out << "0.0 1.0 zlo zhi" << std::endl;



    // blank line
    out << std::endl;

    out << "Masses\n" << std::endl;
    for ( int i=0 ; i<species.size(); i++ ) {
        out << i+1 << " " << species[i].mass << std::endl;
    }


    // blank line
    out << std::endl;


    // Write out particle coordinates and types
    out << "Atoms\n" << std::endl;
    for ( int i=0 ; i<nstot; i++ ) {

        out << i+1 << " " << mID[i]+1 << " " << intSpecies[i]+1 << "  " ;
        for ( int j=0 ; j<Dim ; j++ ) 
            out << x[i*Dim+j] << " " ;
        if ( Dim < 3 )
            out << "0.0 " ;
        out << std::endl;

    }// i=0 ; i<nstot


    // blank line
    out << std::endl;


    // Write out bond list and types
    out << "Bonds\n" << std::endl;
    int bondCounter = 0;
    for ( int i=0 ; i<nstot ; i++ ) {

        for ( int j=0 ; j<nBonds[i]; j++ ) {
            // Only write each bond once (when i < bondedTo)
            if ( i < bondedTo[i*MAXBONDS+j] ) {
                out << bondCounter+1 << " " << bondType[i*MAXBONDS+j]+1 << " " << i+1 << " " << bondedTo[i*MAXBONDS+j]+1 << std::endl;
                
                bondCounter++;
            }
        }
    }


    out.close();
}// end of writeDataConfig


// Reallocates all of the 'particle-size' arrays to the new value of nstot, 'newns'.
// This can also be used for the intial allocation 
// ONLY AFFECTS HOST ARRAYS
void PS_Box::allocHostParticleArrays(int newns) {
    std::cout << "Reallocating for " << newns << " sites on the host..." ;
    x.resize(newns*Dim);
    v.resize(newns*Dim);
    f.resize(newns*Dim);

    intSpecies.resize(newns);

    mID.resize(newns);

    // gridW.resize(newns * gridPerPartic);
    // gridInds.resize(newns * gridPerPartic);

    nBonds.resize(newns);
    bondedTo.resize(newns*MAXBONDS);
    bondType.resize(newns*MAXBONDS);

    nAngles.resize(newns);
    angleGroup.resize(newns*MAXANGLES*3);
    angleType.resize(newns*MAXANGLES);

    std::cout << "done!" << std::endl;
}


// Reallocates all of the 'particle-size' arrays to the new value of nstot, 'newns'.
// This can also be used for the intial allocation 
// ONLY AFFECTS DEVICE ARRAYS
void PS_Box::allocDeviceParticleArrays(int newns) {
    std::cout << "Reallocating for " << newns << " sites on the device..." ;

    if ( d_states != NULL ) {
        cudaFree(d_states);
    }

    cudaMalloc(&d_states, newns * Dim * sizeof(curandState));
    d_initDeviceRNG<<<nsGrid, nsBlock>>>(RANDSEED, d_states, nstot);

    d_x.resize(newns*Dim);
    _d_x = (float*) thrust::raw_pointer_cast(d_x.data());
    
    d_v.resize(newns*Dim);
    _d_v = (float*) thrust::raw_pointer_cast(d_v.data());

    d_f.resize(newns*Dim);
    _d_f = (float*) thrust::raw_pointer_cast(d_f.data());

    d_intSpecies.resize(newns);
    _d_intSpecies = (int*) thrust::raw_pointer_cast(d_intSpecies.data());

    d_mID.resize(newns);
    _d_mID = (int*) thrust::raw_pointer_cast(d_mID.data());

    d_gridW.resize(newns * gridPerPartic);
    _d_gridW = (float*) thrust::raw_pointer_cast(d_gridW.data());

    d_gridInds.resize(newns * gridPerPartic);
    _d_gridInds = (int*) thrust::raw_pointer_cast(d_gridInds.data());

    d_nBonds.resize(newns);
    _d_nBonds = (int*) thrust::raw_pointer_cast(d_nBonds.data());

    d_bondedTo.resize(newns*MAXBONDS);
    _d_bondedTo = (int*) thrust::raw_pointer_cast(d_bondedTo.data());

    d_bondType.resize(newns*MAXBONDS);
    _d_bondType = (int*) thrust::raw_pointer_cast(d_bondType.data());

    
    d_nAngles.resize(newns);
    d_angleGroup.resize(newns*MAXANGLES*3);
    d_angleType.resize(newns*MAXANGLES);

    std::cout << "done!" << std::endl;
}

// Sends all particle-size arrays from host to device. Intended to be used after
// initialization when info needs to go to device for running simulations, though
// could be used any time.
void PS_Box::sendAllHostToDevice(void) {
    d_x = x;
    d_v = v;
    d_f = f;

    for ( int j=0 ; j<Dim ; j++ ) {
        _d_dxf[j] = (float)d_dx[j];
    }

    d_intSpecies = intSpecies;
    
    d_nBonds = nBonds;
    d_bondedTo = bondedTo;
    d_bondType = bondType;

    d_nAngles = nAngles;
}



// Generate a new linear polymer of arbitrary blockiness and add it to the box
void PS_Box::makeLinear(std::istringstream& iss ) {
    if ( rho0 < 0.0 ) die("rho0 must be defined before molecules created!");

    int numBlocks, Ntot = 0;

    // Both set to negative values to determine which keyword given
    double phi = -1.0;  
    int nmolecs = -1; 

    std::string s1;

    // read either phi or nmolecs keyword
    iss >> s1;
    if ( s1 == "phi" ) {
        iss >> phi;
    }
    else if ( s1 == "nmolecs" ) {
        iss >> nmolecs;
    }
    else { 
        std::cout << "invalid keyword: " << s1 << std::endl;
        die("invalid keyword after molecule linear. must be 'phi' or 'nmolecs'"); 
    }

    // Read in the number of blocks
    iss >> numBlocks; 
    
    std::vector<int> Nblocks(numBlocks);
    std::vector<int> blockBondType(numBlocks,0);
    std::vector<int> wlcFlag(numBlocks,0);
    std::vector<int> drudeFlag(numBlocks,0);
    std::vector<int> chargeFlag(numBlocks,0);
    std::vector<std::string> speciesBlocks(numBlocks);
    std::vector<int> intSpeciesBlocks(numBlocks);

    // Store the basic block info
    for (int j=0 ; j<numBlocks; j++ ) {
        iss >> Nblocks[j];
        iss >> speciesBlocks[j];

        Ntot += Nblocks[j];
    }

    // Check for optional arguments
    if ( iss.tellg() != -1 ) {

        iss >> s1;
        if ( s1 == "wlc" || s1 == "drude" || s1 == "charge" ) {
            die("worm-like chains, drude oscillators, and charges not implemented!");
        }
        else if ( s1 == "bondType" || s1 == "bondtype" ) {
            die("bond types not set up in ps_box make linear");
            // make sure to decide on how to handle junction cases, document it
        }
    }

    // Compute number of molecules of this type to add
    // if volume fraction was read
    if ( phi > 0.0 ) { nmolecs = int( rho0 * V * phi / float(Ntot) ); }

    std::cout << "Generating " << nmolecs << " molecules each with " << Ntot << " sites" << std::endl;

    // particle index to be incremented as particles added
    int ind = nstot;


    // Update number of sites in the box
    nstot += nmolecs * Ntot;
    allocHostParticleArrays(nstot);
    std::cout << "nstot changed values to: " << nstot << ", starting index: " << ind << std::endl;



    // Find starting molecule ID
    int molecInd = -1;
    for ( int i=0 ; i<nstot ; i++ ) {
        if (mID[i] > molecInd) molecInd = mID[i];
    }
    if ( molecInd < 0 ) molecInd = 0;




    // Main loop over molecules, blocks, sites on each block
    for ( int i=0 ; i<nmolecs ; i++ ) {
        for ( int j=0 ; j<numBlocks; j++ ) {
            int speciesVal = findSpeciesInteger(speciesBlocks[j]);
            
            for ( int s=0 ; s<Nblocks[j]; s++ ) {

                // Track species info
                intSpecies[ind] = speciesVal;
                
                
                // Is this a chain end? 
                // If so, place randomly in the box
                if ( j==0 && s==0 ) {
                    for ( int a=0 ; a<Dim ; a++ ) {
                        x[ind*Dim+a] = ran2() * L[a];
                    }
                }

                // Not a chain end: place monomer a unit vector away
                // from previous monomer
                else {
                    double ru[3];
                    random_unit_vec(ru, Dim);

                    for ( int a=0 ; a<Dim ; a++ ) {
                        int prevXInd = (ind-1)*Dim+a;
                        int Xind = ind*Dim+a;

                        x[Xind] = x[prevXInd] + ru[a];

                        if ( x[Xind] > L[a] ) x[Xind] -= L[a];
                        else if ( x[Xind] < 0.0 ) x[Xind] += L[a];
                    }
                }

                // Initialize velocities, forces to 0.0
                for ( int a=0 ; a<Dim ; a++ ) {
                    v[ind*Dim+a] = f[ind*Dim+a] = 0.0;
                }

                nBonds[ind] = 0;
                
                // Initialize bonds
                // If not the first monomer on a chain, make a bond to previous monomer
                if ( j != 0 || s != 0 ) {
                    int nb = nBonds[ind];
                    
                    bondedTo[ind*MAXBONDS+nb] = ind-1;
                    bondType[ind*MAXBONDS+nb] = blockBondType[j];
                    nBonds[ind]++;
                }

                // if not the last monomer on a chain, make a bond to next monomer
                if ( j != (numBlocks-1) || s != (Nblocks[j]-1 ) ) {
                    int nb = nBonds[ind];
                    
                    bondedTo[ind*MAXBONDS+nb] = ind+1;
                    bondType[ind*MAXBONDS+nb] = blockBondType[j];
                    nBonds[ind]++;
                }

                mID[ind] = molecInd;

                // Increment the particle index
                ind++;

            }// s=0:N[j]

        }// j=0:numBlocks; 

        // Increment molecule index
        molecInd++;
    }// i=0:nmolecs


    std::cout << "nstot is " << nstot << " after molecule creation" << std::endl;
}

void PS_Box::writeGSDtraj() {

    int i;

    gsd_handle gsd_file; 

    if (totSteps == 0){
        std::vector<unsigned int> types(nstot), molecule_ids(nstot);

        for (i = 0; i < nstot; i++) {
            types[i] = intSpecies[i] + 1;
            molecule_ids[i] = mID[i] + 1;
        }

        std::vector<float> masses(nstot);

        for (i = 0; i < nstot; i++) {
            masses[i] = speciesMass[intSpecies[i]];
        }


        auto version = gsd_make_version(1, 4);
        gsd_create_and_open(&gsd_file, gsd_name.c_str(), "gpu-tild", "hoomd", version, gsd_open_flag::GSD_OPEN_APPEND, 0);

        unsigned int frame = totSteps;
        gsd_write_chunk(
            &gsd_file, "configuration/step", gsd_type::GSD_TYPE_UINT64,
            1, 1, 0, &frame
        );
        
        gsd_write_chunk(
                &gsd_file, "configuration/dimensions", gsd_type::GSD_TYPE_UINT8,
                1, 1, 0, &Dim
                );


        std::vector<float> box = {L[0], L[1], L[2], 0, 0, 0};
        gsd_write_chunk(
                &gsd_file, "configuration/box", gsd_type::GSD_TYPE_FLOAT,
                6, 1, 0, box.data() );

        {
            unsigned int ntypes = nstot;
            gsd_write_chunk(&gsd_file, "particles/N", gsd_type::GSD_TYPE_UINT32, 1, 1, 0, &ntypes);
        }
        gsd_write_chunk(&gsd_file, "particles/mass", gsd_type::GSD_TYPE_FLOAT, masses.size(), 1, 0, masses.data());
        
        // Write the particle types 
        gsd_write_chunk(&gsd_file, "particles/typeid", gsd_type::GSD_TYPE_UINT32, nstot, 1, 0, types.data());

        // Write the particle molecule ids
        gsd_write_chunk(&gsd_file, "log/particles/moleculeid", gsd_type::GSD_TYPE_UINT32, nstot, 1, 0, molecule_ids.data());



        int max_len = 10;
        char* names = (char*) calloc((nTypes+1) * 13,  sizeof(char));

        std::string str("type");
        for (i = 0; i < (nTypes+1); i++) {
            strcpy(names + i * max_len, (str + std::to_string(i)).c_str());
        }


        gsd_write_chunk(&gsd_file, "particles/types", gsd_type::GSD_TYPE_INT8, (nTypes + 1), max_len, 0, names);

        unsigned int N_bonds = nBondsTot;
        // Write the number of bonds
        gsd_write_chunk(&gsd_file, "bonds/N", gsd_type::GSD_TYPE_UINT32, 1, 1, 0, &N_bonds);

        // // Write the number of bond types
        // gsd_write_chunk(&gsd_file, "bond/types", gsd_type::GSD_TYPE_UINT32, 1, 1, 0, &nbond_types);

        // Write the bondids
        gsd_write_chunk(&gsd_file, "bonds/typeid", gsd_type::GSD_TYPE_UINT32, nBondsTot, 1, 0, list_of_bond_type.data());

        // Write the bonds/group
        gsd_write_chunk(&gsd_file, "bonds/group", gsd_type::GSD_TYPE_UINT32, nBondsTot, 2, 0, list_of_bond_partners.data());

        // // Write the number of angle types
        // gsd_write_chunk(&gsd_file, "angle/types", gsd_type::GSD_TYPE_UINT32, 1, 1, 0, &nangle_types);

        // Write the number of angles
        gsd_write_chunk(&gsd_file, "angles/N", gsd_type::GSD_TYPE_UINT32, 1, 1, 0, &nAnglesTot);

        // Write the angleids
        gsd_write_chunk(&gsd_file, "angles/typeid", gsd_type::GSD_TYPE_UINT32, nAnglesTot, 1, 0, list_of_angle_type.data());

        // Write the angles/group
        gsd_write_chunk(&gsd_file, "angles/group", gsd_type::GSD_TYPE_UINT32, nAnglesTot, 3, 0, list_of_angle_partners.data());

    }
    else{
        gsd_open(&gsd_file, gsd_name.c_str(), gsd_open_flag::GSD_OPEN_APPEND);
        unsigned int frame = totSteps;
        gsd_write_chunk(
            &gsd_file, "configuration/step", gsd_type::GSD_TYPE_UINT64,
            1, 1, 0, &frame
        );
        
    }

    // Transfer coordinates from device to host
    x = d_x;

    // Make a copy of positions that can be shifted by Lh
    float* h_ns_float;
    h_ns_float = (float*) malloc(nstot*Dim*sizeof(float));
    if ( h_ns_float == NULL ) die("failed to allocate h_ns_float");

    for (i = 0; i < nstot; i++) {
        for (int j = 0; j < Dim; j++) {
            h_ns_float[i * Dim + j] = x[i * Dim + j];
            h_ns_float[i * Dim + j] -= Lh[j];
        }
    }

    gsd_write_chunk(&gsd_file, "particles/position", gsd_type::GSD_TYPE_FLOAT, nstot, 3, 0, h_ns_float);

    if ( doCharges ) die("Charges not set up yet in write gsd routine");
        // gsd_write_chunk(&gsd_file, "particles/charge", gsd_type::GSD_TYPE_FLOAT, ns, 1, 0, charges);


    gsd_end_frame(&gsd_file);
    gsd_close(&gsd_file);

    free(h_ns_float);


}



// Reads a GSD frame as an input configuration
// This routine assumes the following have been defined:
// all "species" keyword/classes
// any non-default values of MAXBONDS, MAXANGLES
void PS_Box::readGSDtraj(const char* file_name, int frame_num, int process){

    // If process == 0, then we are doing a read_resume
    // If process == 1, then we are doing a read_restart

    int base_index = 0;

    gsd_handle gsd_file;
    int f =	gsd_open(&gsd_file, file_name, gsd_open_flag::GSD_OPEN_READONLY);
    if (f){
        std::cout << "Error opening gsd file" << std::endl;
        exit(1);
    }

    int tmp_frame = gsd_get_nframes(&gsd_file);
    if (tmp_frame < 0){
        die("No frames in the gsd file");
    }

    if (frame_num < 0){
        frame_num = tmp_frame - 1;
    }

    if (frame_num > tmp_frame){
        std::cout << "Frame number is too large" << std::endl;
        std::string str = "Frame number is too large. Max frame number is " + std::to_string(tmp_frame);
        die(str);
    }

    std::cout << frame_num << std::endl;

    // Get the box dimensions
    const gsd_index_entry* chunk_index;
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "configuration/dimensions");
    if (chunk_index == NULL) {

        chunk_index = gsd_find_chunk(&gsd_file, base_index, "configuration/dimensions");
        if (chunk_index == NULL) {
            std::string me = "Error: Could not find the chunk 'configuration/dimensions' in the GSD file.";
            die(me);
        }
    }
    gsd_read_chunk(&gsd_file, &Dim, chunk_index);
    n_P_comps = int(Dim*(Dim+ 1))/2;

    
    // Read in the number of atoms in the box
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "particles/N");
    int tmp_ns = 0;
    if (chunk_index == NULL) {
        chunk_index = gsd_find_chunk(&gsd_file, base_index, "particles/N");
        if (chunk_index == NULL) {
            std::string me = "Error: Could not find the chunk 'particles/N' in the GSD file.";
            die(me);
        }
    }

    gsd_read_chunk(&gsd_file, &tmp_ns, chunk_index);
    if (process == 0 && tmp_ns != nstot){
        std::string me = "Error: The number of atoms in the GSD file does not match the number of atoms in the simulation.";
        die(me);
    }
    else {
        nstot = tmp_ns;
    }

    
    // read the charges of the particles
    std::vector<float> charges_tmp(nstot);
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "particles/charge");
    if (chunk_index != NULL){
        doCharges = true;
        gsd_read_chunk(&gsd_file, charges_tmp.data(), chunk_index);
    }


    // Read in the box size
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "configuration/box");
    if (chunk_index == NULL){
        chunk_index = gsd_find_chunk(&gsd_file, base_index, "configuration/box");
        if (chunk_index == NULL) {
            std::string me = "Error: Could not find the chunk 'configuration/box' in the GSD file.";
            die(me);
        }
    }
    gsd_read_chunk(&gsd_file, &L, chunk_index);

    V = 1;
    for (int i = 0; i < Dim; i++) {
        Lh[i] = L[i] / 2.0;
        V *= L[i];
    }


    // Read in the number of bonds
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "bonds/N");
    if (chunk_index == NULL){
        chunk_index = gsd_find_chunk(&gsd_file, base_index, "bonds/N");
        if (chunk_index == NULL) {
            std::string me = "Error: Could not find the chunk 'bonds/N' in the GSD file.";
            die(me);
        }
    }
    gsd_read_chunk(&gsd_file, &nBondsTot, chunk_index);

    std::cout << "  from GSD, nBondsTot = " << nBondsTot << std::endl;

    // Read in the number of angles
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "angles/N");
    if (chunk_index == NULL){
        chunk_index = gsd_find_chunk(&gsd_file, base_index, "angles/N");
        if (chunk_index == NULL) {
            std::string me = "Error: Could not find the chunk 'angles/N' in the GSD file.";
            die(me);
        }
    }
    gsd_read_chunk(&gsd_file, &nAnglesTot, chunk_index);

    std::cout << "n_total_angles = " << nAnglesTot << std::endl;

    // Read in all the particle types to determine the number of types
    // Instead of sorting, we could just allocate the highest amount that there would be in the box
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "particles/typeid");
    std::vector<int> typeids(nstot), typ_id(nstot);
    typeids.resize(nstot);
    typ_id.resize(nstot);

    if (chunk_index == NULL){
        chunk_index = gsd_find_chunk(&gsd_file, base_index, "particles/typeid");
        if (chunk_index == NULL) {
            std::string me = "Error: Could not find the chunk 'particles/typeid' in the GSD file.";
            die(me);
        }
    }

    gsd_read_chunk(&gsd_file, typeids.data(), chunk_index);

    typ_id = typeids;
    {
        std::sort(typ_id.begin(), typ_id.end());
        int max_val = *max_element(typ_id.begin(), typ_id.end());
        int max_valu2 = std::unique(typ_id.begin(), typ_id.end()) - typ_id.begin();
        nTypes = max(max_valu2, max_val - 1 );
    }

    std::cout << "ntypes = " << nTypes << std::endl;

    // Read in all the particle bonds to determine the number of bonds
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "bonds/typeid");
    std::vector<unsigned int> bonds(nBondsTot);

    if (chunk_index == NULL){
        chunk_index = gsd_find_chunk(&gsd_file, base_index, "bonds/typeid");
        if (chunk_index == NULL) {
            std::string me = "Error: Could not find the chunk 'bonds/typeid' in the GSD file.";
            die(me);
        }
    }

    gsd_read_chunk(&gsd_file, bonds.data(), chunk_index);
    list_of_bond_type = bonds;

    if (nBondsTot > 0) {
        std::sort(bonds.begin(), bonds.end());
        int max_val = *max_element(bonds.begin(), bonds.end());
        int max_valu2 = std::unique(bonds.begin(), bonds.end()) - bonds.begin();
        nBondTypes = max(max_valu2, max_val - 1 );
    }

    std::cout << "nbond_types = " << nBondTypes << std::endl;


    // Read in all the particle angles to determine the number of angles
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "angles/typeid");
    std::vector<unsigned int> angles(nAnglesTot);

    if (chunk_index == NULL){
        chunk_index = gsd_find_chunk(&gsd_file, base_index, "angles/typeid");
        if (chunk_index == NULL) {
            std::string me = "Error: Could not find the chunk 'angles/typeid' in the GSD file.";
            die(me);
        }
    }

    gsd_read_chunk(&gsd_file, angles.data(), chunk_index);
    list_of_angle_type = angles;
    if (nAnglesTot > 0) {
        std::sort(angles.begin(), angles.end());
        int max_val = *max_element(angles.begin(), angles.end());
        int max_valu2 = std::unique(angles.begin(), angles.end()) - angles.begin();
        nAngleTypes = max(max_valu2, max_val - 1 );
    }
    else {
        nAngleTypes = 0;
    }

    std::cout << "nangle_types = " << nAngleTypes << std::endl;

    list_of_angle_type.resize(nAnglesTot);

    // Read in the atoms participating in each bond
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "bonds/group");
    std::vector<unsigned int> bond_partners(nBondsTot*2);

    if (chunk_index == NULL){
        chunk_index = gsd_find_chunk(&gsd_file, base_index, "bonds/group");
        if (chunk_index == NULL) {
            std::string me = "Error: Could not find the chunk 'bonds/group' in the GSD file.";
            die(me);
        }
    }

    gsd_read_chunk(&gsd_file, bond_partners.data(), chunk_index);
    list_of_bond_partners = bond_partners;


    // Read in the atoms participating in each angle
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "angles/group");
    std::vector<unsigned int> angle_partners(nAnglesTot*3);

    if (chunk_index == NULL){
        chunk_index = gsd_find_chunk(&gsd_file, base_index, "angles/group");
        if (chunk_index == NULL) {
            std::string me = "Error: Could not find the chunk 'angles/group' in the GSD file.";
            die(me);
        }
    }

    gsd_read_chunk(&gsd_file, angle_partners.data(), chunk_index);
    list_of_angle_partners = angle_partners;


    // Read in the molecule ids
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "log/particles/moleculeid");
    std::vector<unsigned int> molecule_id(nstot);

    if (chunk_index == NULL){
        chunk_index = gsd_find_chunk(&gsd_file, base_index, "log/particles/moleculeid");
        if (chunk_index == NULL) {
            std::string me = "Error: Could not find the chunk 'log/particles/moleculeid' in the GSD file.";
            die(me);
        }
    }

    // Read in the particle moleculeids
    gsd_read_chunk(&gsd_file, molecule_id.data(), chunk_index);
    auto local = molecule_id;

    {
        std::sort(local.begin(), local.end());
        int max_val = *max_element(local.begin(), local.end());
        int max_valu2 = std::unique(local.begin(), local.end()) - local.begin();
        nMolecules  = max(max_valu2, max_val - 1 );
    }

    std::cout << "n_molecules = " << nMolecules << std::endl;

    // Read in particle masses
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "particles/mass");
    std::vector<float> masses(nstot);

    if (chunk_index == NULL){
        chunk_index = gsd_find_chunk(&gsd_file, base_index, "particles/mass");
        if (chunk_index == NULL) {
            std::string me = "Error: Could not find the chunk 'particles/mass' in the GSD file.";
            die(me);
        }
    }

    gsd_read_chunk(&gsd_file, masses.data(), chunk_index);

    // This commented out by RAR
    // Not clear its needed with masses associated with particle species
    // std::map<int, int> map_of_particle_id_mass;

    // for (int i = 0; i < ns; i++) {
    //     if (map_of_particle_id_mass.find(typeids.at(i)) == map_of_particle_id_mass.end()) {
    //         map_of_particle_id_mass.insert(std::pair<int, int>(typeids.at(i), masses.at(i)));
    //     }
    //     else {
    //         // Check if the masses are the same
    //         if (map_of_particle_id_mass.at(typeids.at(i)) != masses.at(i)) {
    //             std::string me = "Error: The masses of the particles with the same type are not the same.";
    //             die(me);
    //         }
    //     }
    // }



    // This is for reading in configuration file data
    if (process != 0) {
        allocHostParticleArrays(nstot);

        printf("Particle memory allocated on host via GSD!\n");


        // Store the types
        for (int i = 0; i < nstot; i++){
            intSpecies[i] = typeids.at(i) - 1;
        }

        // Store the molecule idsids
        for (int i = 0; i < nstot; i++){
            mID[i] = molecule_id.at(i) - 1;
        }

        // As above, commented out bc maybe not necessary?
        // Assign the masses using the map
        // for (auto& x: map_of_particle_id_mass) {
        //     mass[x.first - 1] = x.second;
        // }


        // Zero out the counters
        for ( int i=0 ; i<nstot; i++ ) {
            nBonds[i] = 0;
            nAngles[i] = 0;
        }

        // store the bonds
        for (unsigned int i = 0;  i<list_of_bond_type.size(); i++){
            int i1 = list_of_bond_partners.at(i*2);
            int i2 = list_of_bond_partners.at(i*2+1);
            int b_type = list_of_bond_type.at(i);

            bondedTo[i1 * MAXBONDS + nBonds[i1]] = i2;
            bondType[i1 * MAXBONDS + nBonds[i1]] = b_type;
            nBonds[i1]++;

            bondedTo[i2 * MAXBONDS + nBonds[i2]] = i1;
            bondType[i2 * MAXBONDS + nBonds[i2]] = b_type;
            nBonds[i2]++;
        }


        // store the angles
        for (unsigned int i = 0; i < list_of_angle_type.size(); i++){
            int i1 = list_of_angle_partners.at(i*3);
            int i2 = list_of_angle_partners.at(i*3+1);
            int i3 = list_of_angle_partners.at(i*3+2);

            int a_type = list_of_angle_type.at(i);

            int na = nAngles[i1];
            angleGroup[i1*MAXANGLES*3 + na*3 + 0] = i1;
            angleGroup[i1*MAXANGLES*3 + na*3 + 1] = i2;
            angleGroup[i1*MAXANGLES*3 + na*3 + 2] = i3;
            angleType[i1*MAXANGLES + na] = a_type;
            nAngles[i1] += 1;

            na = nAngles[i2];
            angleGroup[i2*MAXANGLES*3 + na*3 + 0] = i1;
            angleGroup[i2*MAXANGLES*3 + na*3 + 1] = i2;
            angleGroup[i2*MAXANGLES*3 + na*3 + 2] = i3;
            angleType[i2*MAXANGLES + na] = a_type;
            nAngles[i2] += 1;

            na = nAngles[i3];
            angleGroup[i3*MAXANGLES*3 + na*3 + 0] = i1;
            angleGroup[i3*MAXANGLES*3 + na*3 + 1] = i2;
            angleGroup[i3*MAXANGLES*3 + na*3 + 2] = i3;
            angleType[i3*MAXANGLES + na] = a_type;
            nAngles[i3] += 1;
 
        }
    }


    // Make a copy of positions that can be shifted by Lh
    float* h_ns_float;
    h_ns_float = (float*) malloc(nstot*Dim*sizeof(float));
    if ( h_ns_float == NULL ) die("failed to allocate h_ns_float");

    // read in the position of the particles
    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "particles/position");
    gsd_read_chunk(&gsd_file, h_ns_float, chunk_index);
    if (chunk_index == NULL) {
        std::string me = "error: could not find the chunk 'particles/position' in the gsd file.";
        die(me);
    }

    // Store the positions
    for (int i = 0; i < nstot; i++) {
        for (int j = 0; j < Dim; j++) {
            x[ i * Dim + j] = h_ns_float[i * Dim + j] + Lh[j];
        }
    }

    // if (Charges::do_charges){
    //     for (int idx = 0; idx < charges_tmp.size(); idx++)
    //         charges[idx] = charges_tmp.at(idx);
    // }


    chunk_index = gsd_find_chunk(&gsd_file, frame_num, "configuration/step");

    if (chunk_index == NULL) {
        std::string me = "error: could not find the chunk 'configuration/step' in the gsd file.";
        die(me);
    }

    int tmp_step;
    gsd_read_chunk(&gsd_file, &tmp_step, chunk_index);
    totSteps = tmp_step;

    gsd_close(&gsd_file);
    free(h_ns_float);
}