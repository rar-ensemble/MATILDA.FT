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
        d_dxf, nstot, pmeorder, M, Dim );


    ///////////////////////////
    // UPDATE DENSITY FIELDS //
    ///////////////////////////

    // update the density fields
    for ( int i=0 ; i<psGroup.size(); i++ ) {
        // zero density, grid force fields
        psGroup[i].zeroFields();

        // Fill density field
        psGroup[i].makeDensityField();
    }

    // Zero particle forces
    d_assignFloatVal<<<DnsGrid, nsBlock>>>(_d_f, 0.0, Dim*nstot);



    ////////////////////
    // COMPUTE FORCES //
    ////////////////////
    forces();


    // Second integration step
    for ( int i=0 ; i<integrators.size(); i++ ) {
        integrators[i]->Integrate_2();
    }


    // Write log data
    if ( step % logFreq == 0 ) {
        writeData(step);
    }


    // Write field data
    if ( fieldFreq > 0 && step % fieldFreq == 0 ) {
        writeFields();
    }

} // doTimeStep


void PS_Box::NVT(int maxSteps) {
    std::cout << "RUNNING NVT?!?" << std::endl;
    
    for ( int i=0 ; i<maxSteps; i++ ) {
        doTimeStep(i);

        totSteps++;
    }

    writeData(maxSteps);
    writeFields();

}


// Calls all of the force functions
// Assumes forces have been zeroed and 
// initialized before entering this routine.
void PS_Box::forces() {
    
    // 1. bonded forces; 

    // 2. NB forces; 
    
    // 3. Extras

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


// Currently loops over groups and writes field-based densities
void PS_Box::writeFields() {
    std::cout << "  here: ps_box:writefields" << std::endl;
    for (int i=0 ; i<psGroup.size(); i++ ) {
        psGroup[i].writeDensityField();
    }    
}


// Write field of thrust vectors
void PS_Box::writeFieldTFloat(const char* name, thrust::host_vector<float> dat) {
    int i, j, * nn;
    nn = new int[Dim];
    FILE* otp;
    float* r = new float [Dim];


    otp = fopen(name, "w");

    for (i = 0; i < M; i++) {
        get_rf(i, r);
        unstack2(i, nn);

        for (j = 0; j < Dim; j++)
            fprintf(otp, "%f ", r[j]);

        fprintf(otp, "%1.8e \n", dat[i]);

        if (Dim == 2 && nn[0] == Nx[0] - 1)
            fprintf(otp, "\n");
    }

    fclose(otp);
}

// write field of array vectors
void PS_Box::writeFieldFloat(const char* name, const float* dat) {
    int i, j, * nn;
    nn = new int[Dim];
    FILE* otp;
    float* r = new float [Dim];


    otp = fopen(name, "w");

    for (i = 0; i < M; i++) {
        get_rf(i, r);
        unstack2(i, nn);

        for (j = 0; j < Dim; j++)
            fprintf(otp, "%f ", r[j]);

        fprintf(otp, "%1.8e \n", dat[i]);

        if (Dim == 2 && nn[0] == Nx[0] - 1)
            fprintf(otp, "\n");
    }

    fclose(otp);
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

// Sends all particle-size arrays from host to device. Intended to be used after
// initialization when info needs to go to device for running simulations, though
// could be used any time.
void PS_Box::sendAllHostToDevice(void) {
    d_x = x;
    d_v = v;
    d_f = f;

    for ( int j=0 ; j<Dim ; j++ ) {
        d_dxf[j] = (float)d_dx[j];
    }

    d_intSpecies = intSpecies;
    
    d_nBonds = nBonds;
    d_bondedTo = bondedTo;
    d_bondType = bondType;

    d_bondStyle = bondStyle;
    d_bondK = bondK;
    d_bondReq = bondReq;

    d_nAngles = nAngles;
    d_angleGroup = angleGroup;
    d_angleType = angleType;

    d_angleTheq = angleTheq;
    d_angleK = angleK;
    d_angleStyle = angleStyle;

}



