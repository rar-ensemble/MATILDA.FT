// This file contains all of the routines associated with 
// initializing a simulation box for a particle simulation.
// This includes memory allocation on host and device, 
// reading the input file, computing quantities derived 
// from the input parameters.

#include "PS_Box.h"
#include "random.h"
#include "include_libs.h"
#include "gsd.h"
#include <algorithm>
#include <map>


void die(const char*);
double ran2(void);

__global__ void d_initDeviceRNG(unsigned int, curandState*, int);

Integrator* IntegratorFactory(std::istringstream&, PS_Box*);

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
    nstot = nBondsTot = nAnglesTot = nBondTypes = nAngleTypes = 0;

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
                nBondTypes = bondK.size();
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

            else if ( firstWord == "fieldFreq" || firstWord == "field_freq" ) {
                iss >> fieldFreq;
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
            std::cout << "endBox caught, finishing initialization and sending data to GPU" << std::endl;
            break;
        }

    }// while (!inp.eof()), finished reading up to 'endBox' or end of file


    finishInitialization();
    simTime = time(0);

}// End of readInput()


void PS_Box::initializeSim() {
    

    // Initialize the output stream
    OTP.open("ps_data.dat");
    OTP.close();

    totSteps = 0;
}





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

        Lh[j] = 0.5 * L[j];

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

    GSDinit();
    writeGSDtraj();
    // die("initialization finished, GSD written?");

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



    d_bondStyle.resize(nBondTypes);
    _d_bondStyle = (int*) thrust::raw_pointer_cast(d_bondStyle.data());

    d_bondK.resize(nBondTypes);
    _d_bondK = (float*) thrust::raw_pointer_cast(d_bondK.data());

    d_bondReq.resize(nBondTypes);
    _d_bondReq = (float*) thrust::raw_pointer_cast(d_bondReq.data());

    
    d_nAngles.resize(newns);
    d_angleGroup.resize(newns*MAXANGLES*3);
    d_angleType.resize(newns*MAXANGLES);

    std::cout << "done!" << std::endl;
}