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
__global__ void d_calcGridWeights(float*, int*, const float*, const int*, 
const float*, const int, const int, const int, const int);

Integrator* IntegratorFactory(std::istringstream&, PS_Box*);
PS_Potential* PSPotentialFactory(std::istringstream&, PS_Box*);
PS_Compute* PSComputeFactory(std::istringstream&, PS_Box*);
PS_Group* GroupFactor(std::istringstream&, PS_Box*);

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
    NSEXTRA = 0;
    MAXBONDS = 2;
    MAXANGLES = 3;
    Nr = 100;
    rho0 = C = -1.0;
    nstot = nBondsTot = nAnglesTot = nBondTypes = nAngleTypes = 0;
    idum = RANDSEED = time(0);
    pmeorder = 1;
    verbose = false;
    boxStyle = "ps";
    firstAllocDone = 0;

    // Some default values
    logFreq = 100;
    gsdFreq = 0;
    fieldFreq = 0;
    gsd_name = "traj.gsd";
    trajFileName = "traj.lammpstrj";
    datFileName = "ps_data.dat";
    initDataFileName = "init.input.data";
    trajFreq = 0;
    doCharges = 0;
    logSpaceGSDflag = logSpaceFieldflag = 0;
    logScaleGSD = logScaleFields = 1.0;

    std::string line, firstWord;

    bool readDimension = false;

    while (!inp.eof()) {
        getline(inp, line);

        if ( line.length() == 0 || line.at(0) == '#')
            continue;

        std::istringstream iss(line);
        
        while ( iss >> firstWord ) {
            

            if ( firstWord == "endBox" ) {
                break;
            }

            if ( firstWord == "angle" ) {
                int atype;
                iss >> atype;
                if ( atype > angleK.size() ) {
                    angleK.resize(atype);
                    angleTheq.resize(atype);
                    angleStyle.resize(atype);
                }

                std::string style;
                iss >> style;

                iss >> angleK[atype-1];
                if ( style == "wlc" ) {
                    angleStyle[atype-1] = 0;
                }
                else if ( style == "harmonic" ) {
                    angleStyle[atype-1] = 1;
                    iss >> angleTheq[atype-1];
                    angleTheq[atype-1] *= PI / 180.0;
                }
                else die("Angle style not supported!");

                nAngleTypes = angleK.size();

            }

            else if ( firstWord == "blocksize" || firstWord == "blockSize" ) { iss >> blockSize ; }


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
                else { die("Bond type not supported!"); }
                nBondTypes = bondK.size();
            }

            // Commands are alphabetical from here on
            else if ( firstWord == "boxLengths" ) {
                if (!readDimension) 
                    std::cout << "\nWARNING! WARNING!\nboxLengths read before dimension specified, assuming default value of 2"<<std::endl;

                for ( int j=0 ; j<Dim ; j++ ) {
                    if ( !(iss >> L[j])) { die("Failed to read boxlength value! Did you include values for each dimension?"); }
                    V *= L[j];
                    if ( Nx[0] > 0 ) { dx[j] = L[j] / double(Nx[j]); }
                }
            }


            else if ( firstWord == "compute" ) {
                computes.push_back( PSComputeFactory(iss, this) );
            }

            else if ( firstWord == "datFileName" || firstWord == "dat_file_name" ) {
                iss >> datFileName;
            }

            else if ( firstWord == "Dim" ) {
                iss >> Dim;
                setDimension(Dim);
                readDimension = true;
                if ( Dim == 2 ) n_P_comps = 3;
                else n_P_comps = 6;
            }

            else if ( firstWord == "doCharges" ) {
                this->enableCharges();
            }

            else if ( firstWord == "fieldFreq" || firstWord == "field_freq" ) {
                iss >> fieldFreq;

                if ( iss.tellg() != -1 ) {
                    std::string tps;
                    iss >> tps;
                    if ( tps != "logspace" && tps != "log_space" && tps != "logSpace") {
                        die("PS_Box_init::readInput: invalid option to gsdFreq");
                    }
                    logSpaceFieldflag = 1;
                    iss >> logScaleFields;
                }
            }

            else if ( firstWord == "grid" ) {
                if ( !readDimension ) { die("Dim must be defined before grid!" );}
                for ( int j=0 ; j<Dim ; j++ ) {
                    if ( !(iss >> Nx[j]) ) { die("Failed to read grid value! Did you include values for each dimension?"); }
                    M *= Nx[j];
                    if ( L[0] > 0.0 ) { dx[j] = L[j] / double(Nx[j]); }
                }
            }

            else if ( firstWord == "group") {
                psGroup.push_back( PS_Group(iss,this));
                // species.push_back(PS_Species(iss, this));
            }

            else if ( firstWord == "gsdFreq" || firstWord == "gsd_freq" ) {
                iss >> gsdFreq;

                if ( iss.tellg() != -1 ) {
                    std::string tps;
                    iss >> tps ;
                    if ( tps != "logspace" && tps != "log_space" && tps != "logSpace") {
                        die("PS_Box_init::readInput: invalid option to gsdFreq");
                    }
                    logSpaceGSDflag = 1;
                    iss >> logScaleGSD;
                }
            }

            else if ( firstWord == "gsdName" || firstWord == "gsd_name" ) {
                iss >> gsd_name;
            }

            else if ( firstWord == "integrator" ) {
                integrators.push_back( IntegratorFactory(iss, this) );
            }

            else if ( firstWord == "initDataFileName" ) {
                iss >> initDataFileName;
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
                else if ( nextWord == "grafted") {
                    makeGrafted(iss);
                }
                else if ( nextWord == "sclc" ) {
                    makeSCLC(iss);
                }
                else { die("PS_Box_init:readInput: invalid molecule type read"); }
            }

            else if ( firstWord == "Nr" ) {
                iss >> Nr;
            }

            else if ( firstWord == "NSEXTRA" || firstWord == "nsextra" ) {
                iss >> NSEXTRA;
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

            else if ( firstWord == "potential" ) {
                potentials.push_back( PSPotentialFactory(iss, this) );
            }

            else if ( firstWord == "neighbor_list" ) {
                neighborLists.push_back( new PS_NeighborList(iss, this) );
            }


            else if (firstWord == "randSeed" || firstWord == "RAND_SEED" || firstWord == "RANDSEED") {
                iss >> idum;        // Set CPU RNG to have seed = RANDSEED
                RANDSEED = idum;    // Set GPU RNG to have seed = RANDSEED
            }

            else if ( firstWord == "readData" ) {
                std::string dataName;
                iss >> dataName;
                readDataConfig(dataName.c_str());
            }

            else if (firstWord == "rho0") {
                iss >> rho0;
            }

            else if ( firstWord == "species" ) {
                species.push_back(PS_Species(iss, this));
            }

            else if ( firstWord == "trajFileName" ) {
                iss >> trajFileName;
            }

            else if ( firstWord == "trajFreq" || firstWord == "traj_freq" ) {
                iss >> trajFreq;
            }

            else if ( firstWord == "verbose" ) {
                iss >> verbose;
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
    check_cudaError("end of parsing input");

    finishInitialization();
    simTime = time(0);

}// End of readInput()





// After box is created by input file, this completes the initialization
void PS_Box::finishInitialization() {

    if ( nstot == 0 ) {
        die("Box created with no particles?!?");
    }

    // Initialzie the data file, write its header
    OTP.open(datFileName);
    OTP << "# step" ;
    if ( nBondTypes > 0 ) OTP << " bond" ;
    
    OTP << std::endl;
    OTP.close();

    // After input read, make the FFT plan
    // This currently assumes complex-float to complex-float transforms
    // Would probably be better to do R2C and C2R at some point
    if ( this->Dim == 2 ) 
        cufftPlan2d(&fftplanSingle, Nx[1], Nx[0], CUFFT_C2C);
    if ( this->Dim == 3 ) 
        cufftPlan3d(&fftplanSingle, Nx[2], Nx[1], Nx[0], CUFFT_C2C);

    // Define gvol, dx, gridPerPartic
    gvol = 1.0;
    gridPerPartic = 1;
    for ( int j=0 ; j<Dim ; j++ ) {
        dx[j] = L[j] / double(Nx[j]);
        gvol *= dx[j];

        Lh[j] = 0.5 * L[j];

        gridPerPartic *= (pmeorder+1);
    }

    std::cout << "Computed gridPerPartic = " << gridPerPartic << std::endl;

    // gpuGrid, block sizes
    M_Block = blockSize;
    M_Grid = (int)ceil((double)(M) / M_Block);
    DMGrid = (int)ceil((double)(Dim*M) / M_Block);

    nsBlock = blockSize;
    nsGrid = (int)ceil((double)(nstot) / nsBlock);
    DnsGrid = (int)ceil((double)(Dim*nstot) / nsBlock);
    

    // Ensure charge neutrality
    if (doCharges) {
        float qtot = 0.0, qneg = 0.0, qpos = 0.0;
        for ( int i=0 ; i<nstot; i++ ) {
            qtot += charges[i];
            if ( charges[i] < 0.0 ) qneg += charges[i];
            else if ( charges[i] > 0.0 ) qpos += charges[i];
        } 

        if ( qtot != 0.f ) {
            std::string qerror = "Box net charge is not zero! qtotal: " + std::to_string(qtot);
            qerror += "total negative: " + std::to_string(qneg) + ", positive: " + std::to_string(qpos);
            die(qerror);
        }
    }
    
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
        // std::cout << " i, nBonds, nAngs: " << i << " " << nBonds[i] << " " << nAngles[i] << std::endl;
        for ( int j=0 ; j<nBonds[i] ; j++ ) {
            if ( bondedTo[i*MAXBONDS+j] > i ) nBondsTot++;
        }

        for ( int j=0 ; j<nAngles[i] ; j++ ) {
            // only count angle if i == middle particle 
            int aind = 3*(i*MAXANGLES+j);
            if ( angleGroup[aind + 1] == i ) {
                nAnglesTot++;
            }
        }
    }
    

    writeDataConfig(initDataFileName);
    std::cout << "Initial config in data file format written to init.input.data" << std::endl;

    // Finish memory allocation on host
    allocHostParticleArrays(nstot);

    // Allocate device memory and copy device vars
    allocDeviceArrays(nstot + NSEXTRA);
    check_cudaError("allocating device particle arrays");
    
    createDefaultGroups();
    for ( int i=0 ; i<psGroup.size(); i++ ) {
        if ( psGroup[i].nsites == 0 ) {
            die("A group with size 0 found, exiting!");
        }
    }


    // Assign groups to integrators
    for ( int i=0 ; i<integrators.size(); i++ ) {
        integrators[i]->findGroup();
    }

    for ( int i=0 ; i<neighborLists.size(); i++ ) {
        neighborLists[i]->initializeNList();
    }

    // Complete initialization of species variables
    nTypes = species.size();

    speciesMass = (float*) calloc(nTypes, sizeof(float));
    speciesMobility = (float*) calloc(nTypes, sizeof(float));

    // speciesMass.resize(nTypes);
    // speciesMobility.resize(nTypes);
    for ( int i=0 ; i<nTypes; i++ ) {
        speciesMass[i] = species[i].mass;
        speciesMobility[i] = species[i].mobility;
        if ( verbose ) { 
            std::cout << "  type index: " << i << " mass: " << speciesMass[i] << 
            " mobility: " << speciesMobility[i] << std::endl;
        }
    }

    // temp storage arrays
    gabe = (float*) calloc(M, sizeof(float));
    alex = (float*) calloc(M, sizeof(float));
    cpxGabe = (std::complex<float>*) calloc(M, sizeof(std::complex<float>));
    cpxAlex = (std::complex<float>*) calloc(M, sizeof(std::complex<float>));

    cudaMalloc(&d_speciesMass, nTypes*sizeof(float));
    cudaMalloc(&d_speciesMobility, nTypes*sizeof(float));

    cudaMemcpy(d_speciesMass, speciesMass, nTypes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_speciesMobility, speciesMobility, nTypes * sizeof(float), cudaMemcpyHostToDevice);

    check_cudaError("copying species data to device");

    // Send all data to the device
    std::cout << "sending data to device..." ; fflush(stdout);
    sendAllHostToDevice();
    std::cout << "done!" << std::endl;
    check_cudaError("Sending data to device");

    for ( int i=0 ; i<integrators.size() ; i++ ) {
        integrators[i]->finishInitialization();
    }

    for ( int i=0 ; i<potentials.size() ; i++ ) {
        potentials[i]->initializePotential();
    }
    
    // die("potentials written?");


    // INITIALIZE BINARY FILES
    GSDinit();
    writeGSDtraj();
    // die("initialization finished, GSD written?");


    // init grid binary data files
    std::cout << "Initializing binary files..." ;
    for ( int i=0 ; i<psGroup.size() ; i++ ) {
        std::cout << psGroup[i].returnName() << "..." ; fflush(stdout);
        std::string nm = std::string("density-") + psGroup[i].returnName() + std::string(".bin");
        initBinaryDataFile(nm);
    }
    std::cout << "done!" << std::endl;


    for ( int i=0 ; i<potentials.size(); i++ ) {
        potentials[i]->initBinaryOutput();
    }


    // Initialize the output stream
    OTP.open(datFileName);
    OTP.close();

    totSteps = 0;

    
    // Compute grid weights and fill the grid as a final 
    // step before leaving initialization.
    d_calcGridWeights<<<nsGrid, nsBlock>>>(d_gridW, d_gridInds, d_x, _d_Nx, 
            d_dxf, nstot, pmeorder, M, Dim );
    
    for ( int i=0 ; i<psGroup.size(); i++ ) {
        // zero density, grid force fields
        psGroup[i].zeroFields();
        // Fill density fields
        psGroup[i].makeDensityField();
    }
    check_cudaError("density field generation in PS_Box::finishInit");


    if ( verbose ) {
        std::cout << "\n" ;
        std::cout << "WARNING: VERBOSE MODE ENABLED" << std::endl;
        std:: cout << "This will slow the code down but can be helpful for debugging.\n" << std::endl;
    }

}


// Creates groups for each particle type and 'all'
void PS_Box::createDefaultGroups() {
    psGroup.push_back(PS_Group("all", -1, this));

    for ( int i=0 ; i<species.size(); i++ ) {
        psGroup.push_back(PS_Group("type", i, this));

        std::cout << "Group name: " << psGroup[i+1].returnName() << std::endl;
    }

    if ( doCharges ) {
        psGroup.push_back(PS_Group("charges", -1, this));
    }

    std::cout << "Groups for all, each type created" << std::endl;
}


// Reallocates all of the 'particle-size' arrays to the new value of nstot, 'newns'.
// This can also be used for the intial allocation 
// ONLY AFFECTS HOST ARRAYS
void PS_Box::allocHostParticleArrays(int newns) {
    std::cout << "  (Re)allocating for " << newns << " sites on the host..." ; 
    
    x.resize(newns*Dim);
       
    
    // Initial allocation needs to be done with malloc
    if ( firstAllocDone == 0 ) { 
        std::cout << " first allocation using malloc..." ;
        v = (float*) malloc(newns*Dim*sizeof(float)); 
        f = (float*) malloc(newns*Dim*sizeof(float)); 

        nBonds = (int*) malloc(newns * sizeof(int));
        bondedTo = (int*) malloc(newns * MAXBONDS * sizeof(int));
        bondType = (int*) malloc(newns * MAXBONDS * sizeof(int));

        nAngles =    (int*) malloc(newns * sizeof(int));
        angleType =  (int*) malloc(newns * MAXANGLES * sizeof(int));
        angleGroup = (int*) malloc(newns * 3 * MAXANGLES * sizeof(int));

        if ( doCharges ) {
            charges = (float*) malloc( newns * sizeof(float));
        }
    }
    
    // subsequent resizing done with realloc
    else { 
        v = (float*) realloc(v, newns * Dim * sizeof(float)); 
        f = (float*) realloc(v, newns * Dim * sizeof(float)); 

        nBonds = (int* ) realloc(nBonds, newns * sizeof(int));
        bondedTo = (int* ) realloc(bondedTo, newns*MAXBONDS * sizeof(int));
        bondType = (int* ) realloc(bondType, newns*MAXBONDS * sizeof(int));
                
        nAngles =    (int*) realloc(nAngles,    newns*sizeof(int));
        angleType =  (int*) realloc(angleType,  newns*MAXANGLES*sizeof(int));
        angleGroup = (int*) realloc(angleGroup, newns*MAXANGLES*3*sizeof(int));

        if ( doCharges ) {
            charges = (float*) realloc(charges, newns*sizeof(float));
        }

    }


    intSpecies.resize(newns);
    mID.resize(newns);

    firstAllocDone = 1;
    std::cout << "done!" << std::endl;
}


// Reallocates all of the 'particle-size' arrays to the new value of nstot, 'newns'.
// NOTE: This can also be used for the intial allocation 
// NOTE: Data stored in these arrays will be lost
// ONLY AFFECTS DEVICE ARRAYS
void PS_Box::allocDeviceArrays(const int nsAlloc) {
    std::cout << "Allocating for " << nsAlloc << " sites on the device..." ;
    fflush(stdout);

    // if ( d_states != NULL ) {
    //     cudaFree(d_states);
    // }

    cudaMalloc(&d_states, nsAlloc * Dim * sizeof(curandState));
    d_initDeviceRNG<<<nsGrid, nsBlock>>>(RANDSEED, d_states, nstot);

    cudaMalloc(&d_dxf, Dim * sizeof(float));

    cudaMalloc(&d_x, nsAlloc * Dim * sizeof(float));
    cudaMalloc(&d_v, nsAlloc * Dim * sizeof(float));
    cudaMalloc(&d_f, nsAlloc*Dim*sizeof(float));

    
    if ( doCharges ) {
        cudaMalloc(&d_charges, nsAlloc*sizeof(float));
    }

    cudaMalloc(&d_intSpecies, nsAlloc * sizeof(int));
    cudaMalloc(&d_mID, nsAlloc * sizeof(int));


    cudaMalloc(&d_gridW,    nsAlloc*gridPerPartic * sizeof(float));
    cudaMalloc(&d_gridInds, nsAlloc*gridPerPartic * sizeof(int));

    cudaMalloc(&d_nBonds, nsAlloc*sizeof(int));
    cudaMalloc(&d_bondedTo, nsAlloc*MAXBONDS*sizeof(int));
    cudaMalloc(&d_bondType, nsAlloc*MAXBONDS*sizeof(int));

    cudaMalloc(&d_bondStyle,nBondTypes * sizeof(int));
    cudaMalloc(&d_bondReq,  nBondTypes * sizeof(int));
    cudaMalloc(&d_bondK,    nBondTypes * sizeof(int));

    cudaMalloc(&d_nAngles,    nsAlloc*sizeof(int));
    cudaMalloc(&d_angleGroup, nsAlloc*MAXANGLES*3*sizeof(int));
    cudaMalloc(&d_angleType,  nsAlloc*MAXANGLES*sizeof(int));

    cudaMalloc(&d_angleTheq,  nAngleTypes*sizeof(int));
    cudaMalloc(&d_angleK,     nAngleTypes*sizeof(int));
    cudaMalloc(&d_angleStyle, nAngleTypes*sizeof(int));
    

    // Grid-based arrays
    cudaMalloc(&d_Gabe, M * sizeof(float));
    cudaMalloc(&d_Alex, M * sizeof(float));
    cudaMalloc(&d_cpxGabe, M * sizeof(cuComplex));
    cudaMalloc(&d_cpxAlex, M * sizeof(cuComplex));

    // Pre-allocated scratch arrays for computeThermoProps()
    cudaMalloc(&d_thermoE,        nsAlloc * sizeof(float));
    cudaMalloc(&d_bondVirScratch, nsAlloc * n_P_comps * sizeof(float));
    cudaMalloc(&d_angleVirScratch,nsAlloc * n_P_comps * sizeof(float));

    std::cout << "done!" << std::endl;
}


// Sends all particle-size arrays from host to device. Intended to be used after
// initialization when info needs to go to device for running simulations, though
// could be used any time.
void PS_Box::sendAllHostToDevice(void) {
    
    float *xtmp;
    xtmp = (float*) calloc( nstot*Dim, sizeof(float));

    for ( int i=0; i<nstot*Dim; i++ ) {
        xtmp[i] = x[i];
    }

    // Copy positions to device
    cudaMemcpy(d_x, xtmp, nstot*Dim * sizeof(float), cudaMemcpyHostToDevice);
    check_cudaError("positions sent to device");

    float dxf[3];
    if ( Dim > 3 ) die("Dim greater than 3?!??");
    for ( int j=0 ; j<Dim ; j++ ) {
        dxf[j] = (float)dx[j];
    }
    cudaMemcpy(d_dxf, dxf, Dim*sizeof(float), cudaMemcpyHostToDevice);
    check_cudaError("dx box information sent to device");

    cudaMemcpy(d_L, L, Dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Lh, Lh, Dim*sizeof(float), cudaMemcpyHostToDevice);
    d_Nx = Nx;
    check_cudaError("box information sent to device");
    
    sendThrustVectorToDeviceArray(intSpecies, d_intSpecies, nstot);
    sendThrustVectorToDeviceArray(mID, d_mID, nstot);
    check_cudaError("mID and intspecies sent using template");

    cudaMemcpy(d_nBonds, nBonds, nstot * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bondedTo, bondedTo, nstot*MAXBONDS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bondType, bondType, nstot*MAXBONDS * sizeof(int), cudaMemcpyHostToDevice);


    sendThrustVectorToDeviceArray(bondReq, d_bondReq, nBondTypes);
    sendThrustVectorToDeviceArray(bondK, d_bondK, nBondTypes);
    sendThrustVectorToDeviceArray(bondStyle, d_bondStyle, nBondTypes);

    check_cudaError("template sent bond info to device");


    // cudaMemcpy(d_nAngles, nAngles, nstot * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bondedTo, bondedTo, nstot*MAXBONDS * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bondType, bondType, nstot*MAXBONDS * sizeof(int), cudaMemcpyHostToDevice);

    cudaMemcpy(d_nAngles,    nAngles,     nstot*sizeof(int),             cudaMemcpyHostToDevice);
    cudaMemcpy(d_angleType,  angleType,   nstot*MAXANGLES*sizeof(int),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_angleGroup, angleGroup,  nstot*3*MAXANGLES*sizeof(int), cudaMemcpyHostToDevice);
    
    sendThrustVectorToDeviceArray(angleTheq,  d_angleTheq,  nAngleTypes);
    sendThrustVectorToDeviceArray(angleK,     d_angleK,     nAngleTypes);
    sendThrustVectorToDeviceArray(angleStyle, d_angleStyle, nAngleTypes);
    

    if (doCharges) {
        cudaMemcpy(d_charges, charges, nstot*sizeof(float), cudaMemcpyHostToDevice);
    }

    check_cudaError("template sending angle info to device");


    free(xtmp);

}



// Reads input.data style configuration
// Assumes this is the first set of particles allocated -
//  - prob can relax this assumption by just changing below 
//  so that nstot is incremented, and ind initialized from nonzero
// Initial code base copied from v1 read_charge_config routine
void PS_Box::readDataConfig(std::string inpName) {

    if ( nstot != 0 ) {
        die("Must read data file before making additional molecules");
    }


    int i, di, ind, ltp;
	float dx, dy;
	char tt[120];

	FILE* inp;
	inp = fopen(inpName.c_str(), "r");
	if (inp == NULL) {
        std::string death = "Failed to open " + inpName;
		die(death);
	}

	(void)!fgets(tt, 120, inp);
	(void)!fgets(tt, 120, inp);

	(void)!fscanf(inp, "%d", &nstot);      (void)!fgets(tt, 120, inp);
	(void)!fscanf(inp, "%d", &nBondsTot);  (void)!fgets(tt, 120, inp);
	(void)!fscanf(inp, "%d", &nAnglesTot);  (void)!fgets(tt, 120, inp);


	(void)!fgets(tt, 120, inp);

	(void)!fscanf(inp, "%d", &nTypes);  (void)!fgets(tt, 120, inp);
	(void)!fscanf(inp, "%d", &nBondTypes);  (void)!fgets(tt, 120, inp);
	(void)!fscanf(inp, "%d", &nAngleTypes);  (void)!fgets(tt, 120, inp);


	// Read in box shape
	(void)!fgets(tt, 120, inp);
	(void)!fscanf(inp, "%f %f", &dx, &dy);   (void)!fgets(tt, 120, inp);
	L[0] = (float)(dy - dx);
	(void)!fscanf(inp, "%f %f", &dx, &dy);   (void)!fgets(tt, 120, inp);
	L[1] = (float)(dy - dx);
	(void)!fscanf(inp, "%f %f", &dx, &dy);   (void)!fgets(tt, 120, inp);
	if (Dim > 2)
		L[2] = (float)(dy - dx);
	else
		L[2] = 1.0f;

	V = 1.0f;
	for (i = 0; i < Dim; i++) {
		Lh[i] = 0.5f * L[i];
		V *= L[i];
	}
    std::cout << "about to allocate for " << nstot <<  std::endl;


	// Allocate memory for positions //
	allocHostParticleArrays(nstot);

    std::cout << "Particle memory allocated on host in readDataConfig!" << std::endl;

	(void)!fgets(tt, 120, inp);

	// Read in particle masses
	(void)!fgets(tt, 120, inp); // Masses keyword, presumably
	(void)!fgets(tt, 120, inp); // blank line
	for (i = 0; i < nTypes; i++) {
		(void)!fscanf(inp, "%d %f", &di, &dx); (void)!fgets(tt, 120, inp);
		species[di-1].mass = float(dx);
        //speciesMass[di - 1] = float(dx);
	}
	(void)!fgets(tt, 120, inp); // blank line


	// Read in atomic positions
	(void)!fgets(tt, 120, inp); // Atom keyword, presumably
	(void)!fgets(tt, 120, inp); // blank line

	for (i = 0; i < nstot; i++) {
		if (feof(inp)) die("Premature end of input.conf!");

		(void)!fscanf(inp, "%d %d %d", &ind, &di, &ltp);
		ind -= 1; // switch to 0 indexing

        mID[ind] = di - 1;

		intSpecies[ind] = ltp - 1;

        if ( doCharges ) {
            float dcharge;
		    (void)!fscanf(inp, "%f", &dcharge);
		    charges[ind] = dcharge;
        }

		for (int j = 0; j < Dim; j++) {
			(void)!fscanf(inp, "%f", &dx);
			x[ind*Dim+j] = dx;
		}

		(void)!fgets(tt, 120, inp);
	}
	(void)!fgets(tt, 120, inp);

	// Read in bond information
	(void)!fgets(tt, 120, inp);
	(void)!fgets(tt, 120, inp);

	for (i = 0; i < nstot; i++)
		nBonds[i] = 0;

	list_of_bond_partners.reserve(nBondsTot*2);
	list_of_bond_type.reserve(nBondsTot);

	for (i = 0; i < nBondsTot; i++) {
		(void)!fscanf(inp, "%d", &di); // Bond counter
		(void)!fscanf(inp, "%d", &di); // bond type
		int b_type = di - 1; // --> 0 indexing

		(void)!fscanf(inp, "%d", &di); // particle involved in bond
		int i1 = di - 1; // --> 0 indexing

		(void)!fscanf(inp, "%d", &di); // particle involved in bond
		int i2 = di - 1; // --> 0 indexing

		if (i2 < i1) {
			di = i2;
			i2 = i1;
			i1 = di;
		}

		bondedTo[i1*MAXBONDS+nBonds[i1]] = i2;
		bondType[i1*MAXBONDS+nBonds[i1]] = b_type;
		nBonds[i1]++;

		bondedTo[i2*MAXBONDS+nBonds[i2]] = i1;
		bondType[i2*MAXBONDS+nBonds[i2]] = b_type;
		nBonds[i2]++;

		list_of_bond_type.push_back(b_type);
		list_of_bond_partners.push_back(i1);
		list_of_bond_partners.push_back(i2);

	}
	(void)!fgets(tt, 120, inp);


	list_of_angle_partners.reserve(nAnglesTot*3);
	list_of_angle_type.reserve(nAnglesTot);

	// Read in angle information
	for (i = 0; i < nstot; i++)
		nAngles[i] = 0;

	(void)!fgets(tt, 120, inp);
	(void)!fgets(tt, 120, inp);
	for (i = 0; i < nAnglesTot; i++) {

		(void)!fscanf(inp, "%d", &di);
		(void)!fscanf(inp, "%d", &di);

		int a_type = di - 1;

		(void)!fscanf(inp, "%d", &di);
		int i1 = di - 1;

		(void)!fscanf(inp, "%d", &di);
		int i2 = di - 1;

		(void)!fscanf(inp, "%d", &di);
		int i3 = di - 1;

		if (i3 < i1) {
			di = i3;
			i3 = i1;
			i1 = di;
		}


        // Store angle info on particle i1
		int na = nAngles[i1];
        angleGroup[i1*MAXANGLES*3 + 3*na + 0] = i1;
        angleGroup[i1*MAXANGLES*3 + 3*na + 1] = i2;
        angleGroup[i1*MAXANGLES*3 + 3*na + 2] = i3;
		angleType[i1*MAXANGLES + na] = a_type;
		nAngles[i1] += 1;

        // Store angle info particle i2
        na = nAngles[i2];
        angleGroup[i2*MAXANGLES*3 + 3*na + 0] = i1;
        angleGroup[i2*MAXANGLES*3 + 3*na + 1] = i2;
        angleGroup[i2*MAXANGLES*3 + 3*na + 2] = i3;
		angleType[i2*MAXANGLES + na] = a_type;
		nAngles[i2] += 1;

        // Store angle info particle i2
        na = nAngles[i3];
        angleGroup[i3*MAXANGLES*3 + 3*na + 0] = i1;
        angleGroup[i3*MAXANGLES*3 + 3*na + 1] = i2;
        angleGroup[i3*MAXANGLES*3 + 3*na + 2] = i3;
		angleType[i3*MAXANGLES + na] = a_type;
		nAngles[i3] += 1;


		(void)!fgets(tt, 120, inp);

		list_of_angle_type.push_back(a_type);
		list_of_angle_partners.push_back(i1);
		list_of_angle_partners.push_back(i2);
		list_of_angle_partners.push_back(i3);

	}
	fclose(inp);

}


void PS_Box::enableCharges() {
    if ( !doCharges ) {
        doCharges = 1;
        if ( nstot > 0 ) { 
            allocHostParticleArrays(nstot);
            for ( int i=0 ; i<nstot; i++ ) {
                charges[i] = 0.0;
            }
        }
    }    
}