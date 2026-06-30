// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).

#include "PS_Box.h"
#include "random.h"
#include "include_libs.h"
#include "gsd.h"
#include <algorithm>
#include <map>
#include <cstdint>
#include <cstring>

void die(const char*);
double ran2(void);
void random_unit_vec(double*, int);


__global__ void d_calcGridWeights(float*, int*, const float*, const int*, 
const float*, const int, const int, const int, const int);
__global__ void d_bonds(float*, const int*, const int*, const int*,
const float*, const float*, const int*, const float*, const float*,
const float*, const int, const int, const int);
__global__ void d_bondStressEnergy(float*, float*, const float*, 
const int*, const int*, const int*,
const float*, const float*, const int*, const float*, const float*,
const int, const int, const int, const int);
__global__ void d_anglesStressEnergy(float*, float*, const float*, 
const float*, const float*, const int*, const int*, const int*, const int*,
const float*, const float*, const int, const int, const int, const int);

__global__ void d_angles(float*, const float*, const float*, const float*,
const int*, const int*, const int*, const int*, const float*, const float*,
const int, const int, const int);

__global__ void sumArrayKernel(float*, float*, int);

// Diagnostic: copy one stride-Dim component of d_f into a contiguous array
// so sumDeviceArray can reduce it.  Used by logNetForce().
__global__ void d_extractForceComp(
    float* out,             // [nstot] contiguous output
    const float* f,         // [Dim*nstot] forces
    const int dir,          // component to extract (0, 1, or 2)
    const int Dim,
    const int N             // nstot
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N) return;
    out[id] = f[id * Dim + dir];
}


// Diagnostic: copy one stride-Dim component of d_f into a contiguous array
// so sumDeviceArray can reduce it.  Used by logNetForce().
__global__ void d_extractCpxForceComp(
    cuComplex* out,             // [nstot] contiguous output
    const cuComplex* f,         // [Dim*nstot] forces
    const int dir,          // component to extract (0, 1, or 2)
    const int Dim,
    const int N             // nstot
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N) return;
    out[id] = f[id * Dim + dir];
}



Integrator* IntegratorFactory(std::istringstream&, PS_Box*);

// Executes the commands for a given time step
// Updates all fields, then recomputes all molecule densities
// then populating species densities
void PS_Box::doTimeStep(int step) {

    if ( verbose ) std::cout << "in time step " << step << "..." << std::endl;
    // First integration step, when needed (e.g., velo Verlet)
    for ( int i=0 ; i<integrators.size(); i++ ) {
        if ( verbose ) { std::cout << "Integrate 1..." ; fflush(stdout); }
        integrators[i]->Integrate_1();
        if ( verbose ) { std::cout << "done!" << std::endl; }
        // cudaDeviceSynchronize() required here: check_cudaError alone only catches
        // launch errors, not async execution errors.  Without the sync, execution
        // errors from Integrate_1 stay pending and are falsely attributed to Thrust's
        // sort_by_key in the neighbor list build below.
        cudaDeviceSynchronize();
        check_cudaError("Integration step 1");
    }


    // Rebuild neighbor lists
    for ( int i=0 ; i<neighborLists.size(); i++ )
        neighborLists[i]->build();

    // Update grid weights
    if ( verbose ) { cudaDeviceSynchronize(); std::cout << \
        "calcing weights..." << std::endl; }
    d_calcGridWeights<<<nsGrid, nsBlock>>>(d_gridW, d_gridInds, d_x, _d_Nx,
        d_dxf, nstot, pmeorder, M, Dim );
    check_cudaError("Weights calculated in PS_Box");



    ///////////////////////////
    // UPDATE DENSITY FIELDS //
    ///////////////////////////

    // update the density fields
    for ( int i=0 ; i<psGroup.size(); i++ ) {
        // zero density, grid force fields
        psGroup[i].zeroFields();

        // Fill density fields
        psGroup[i].makeDensityField();
    }



    // Zero particle forces
    if ( verbose ) { cudaDeviceSynchronize(); std::cout << \
        "zero particle forces..." ; fflush(stdout); }
    
    d_assignFloatVal<<<DnsGrid, nsBlock>>>(d_f, 0.0, Dim*nstot);
    
    check_cudaError("Particle forces zeroed");
    if ( verbose ) std::cout << "done!" << std::endl;


    
    
    ////////////////////
    // COMPUTE FORCES //
    ////////////////////
    forces();

    // Diagnostic: net force on COM should be exactly zero every step.
    // Any nonzero value identifies the potential/bond/angle contributing drift.
    if ( verbose && (step % logFreq == 0) ) { logNetForce(step); }


    // Second integration step
    for ( int i=0 ; i<integrators.size(); i++ ) {
        if ( verbose ) { cudaDeviceSynchronize(); std::cout << "integrate 2..." ; fflush(stdout); }
        integrators[i]->Integrate_2();
        if ( verbose ) { cudaDeviceSynchronize(); std::cout << "done!" << std::endl; }
    }


    //////////////////
    // Run computes //
    //////////////////
    for ( int i=0 ; i<computes.size(); i++ ) {
        computes[i]->do_compute(step);
    }





    ///////////////
    // I/O steps //
    ///////////////

    // Write log data
    int startTime = time(0);
    if ( step % logFreq == 0 ) {
        if ( verbose ) { std::cout << "log..." ; fflush(stdout); }
        writeData(step);
        if ( verbose ) std::cout << "done!" << std::endl;

        for ( int i=0 ; i<computes.size(); i++ ) {
            computes[i]->write_output();
        }
    }

    // write to GSD traj file
    if ( gsdFreq > 0 && step % gsdFreq == 0 ) { 
        if ( verbose ) { std::cout << "gsd..." ; fflush(stdout); }
        writeGSDtraj();
        if ( verbose ) std::cout << "done!" << std::endl;
    }

    if ( trajFreq > 0 && step % trajFreq == 0 ) {
        if ( verbose ) { std::cout << "lammpstrj..." ; fflush(stdout); }
        writeLammpsTraj(step);
        if ( verbose ) { cudaDeviceSynchronize(); std::cout << "done!" << std::endl; }
    }

    // Write field data
    if ( fieldFreq > 0 && step % fieldFreq == 0 ) {
        writeFields();
    }
    ioTimer += time(0) - startTime;

} // doTimeStep


void PS_Box::NVT(int maxSteps) {
    std::cout << "RUNNING NVT?!?" << std::endl;
    

    std::cout << "Initial values:" << std::endl;
    writeData(-1);



    // Primary loop over steps
    for ( int i=0 ; i<maxSteps; i++ ) {
        doTimeStep(i);

        totSteps++;
    }
    std::cout << "NVT Finished, writing final output" << std::endl;
    writeData(maxSteps);
    writeFields();
    writeGSDtraj(); 
    writeDataConfig("final.input.data");

    writeTime();
    giveQuote();
}


// Calls all of the force functions
// Assumes forces have been zeroed and 
// initialized before entering this routine.
void PS_Box::forces() {

    // 1. bonded forces;
    if ( verbose ) { std::cout << "Into bonds..." ; fflush(stdout); }
    
    d_bonds<<<nsGrid, nsBlock>>>(d_f, d_nBonds, d_bondedTo, d_bondType, d_bondReq, d_bondK,
        d_bondStyle, d_x, d_L, d_Lh, nstot, MAXBONDS, Dim);
    
    if ( verbose ) std::cout << "bonds done, " << std::endl;


    if ( verbose ) { std::cout << "Into angles..." ; fflush(stdout); }

    d_angles<<<nsGrid, nsBlock>>>(d_f, d_x, d_angleK, d_angleTheq, d_angleStyle, 
        d_nAngles, d_angleType, d_angleGroup, d_L, d_Lh, nstot, MAXANGLES, Dim);

    if ( verbose) std::cout << "angles done!" << std::endl;




    // 2a. Grid forces
    for ( int i=0 ; i<potentials.size(); i++ ) {
        potentials[i]->CalcForces();
    }
    

    // 2b. Grid forces --> particle forces
    for ( int i=0 ; i<psGroup.size() ; i++ ) {
        psGroup[i].mapForces();
    }





    // 3. Extras
    // TBD
}


// Computes the total potential energy of the system
void PS_Box::computeThermoProps() {
    // Computes the energy and virial for each particle
    d_bondStressEnergy<<<nsGrid, nsBlock>>>(d_thermoE, d_bondVirScratch,
        d_x, d_nBonds, d_bondedTo, d_bondType, d_bondReq, d_bondK,
        d_bondStyle, d_L, d_Lh, nstot, MAXBONDS, n_P_comps, Dim);

    // Sums over the particle energies.
    // Prefactor 0.5 corrects for double-counting
    Ubond = 0.5 * sumDeviceArray(d_thermoE, blockSize, nstot);

    d_anglesStressEnergy<<<nsGrid, nsBlock>>>(d_thermoE, d_angleVirScratch,
        d_x, d_angleK, d_angleTheq, d_angleStyle, d_nAngles,
        d_angleType, d_angleGroup, d_L, d_Lh, nstot, MAXANGLES, n_P_comps, Dim);

    // Sums over the particle energies
    // Division by 3 corrects triple-counting
    Uangle = sumDeviceArray(d_thermoE, blockSize, nstot) / 3.0;

    Upe = 0.0;
    for ( int i=0 ; i<potentials.size(); i++ ) {
        Upe += potentials[i]->CalcEnergy();
    }

    Upe += Ubond;
    Upe += Uangle;
}


// Sums d_f per dimension and prints.  Called after forces() in doTimeStep.
// For a conservative force field with Newton's 3rd law satisfied, each
// component should be exactly zero (to float round-off).
void PS_Box::logNetForce(int step) {
    float netF[3] = {0.0f, 0.0f, 0.0f};

    for ( int j=0 ; j<Dim ; j++ ) {
        d_extractForceComp<<<nsGrid, nsBlock>>>(d_thermoE, d_f, j, Dim, nstot);
        netF[j] = sumDeviceArray(d_thermoE, blockSize, nstot);
    }
    check_cudaError("logNetForce");

    std::cout << "step " << step << " netF =";
    for ( int j=0 ; j<Dim ; j++ ) std::cout << " " << netF[j];
    std::cout << std::endl;
}


// Write Hamiltonian terms to output file
void PS_Box::writeData(int step) {

    computeThermoProps();

    OTP.open(datFileName, std::ios_base::app);
    std::string outline;

    OTP << step ;
    std::cout << "step: " << step ;

    std::cout << " Upe: " << Upe ;

    if ( nBondsTot > 0 ) {
        OTP << " " << Ubond ;
        std::cout << " Ubond: " << Ubond;
    }
    
    if ( nAnglesTot > 0 ) { 
        OTP << " " << Uangle ;
        std::cout << " Uangle: " << Uangle;
    }

    for ( int i=0 ; i<potentials.size(); i++ ) {
        OTP << " " << potentials[i]->energy;
        std::cout << " pot[" << i << "]: " << potentials[i]->energy;
    }

    OTP << std::endl;
    std::cout << std::endl;

    OTP.close();
}




// Currently loops over groups and writes field-based densities
void PS_Box::writeFields() {
    if ( verbose ) std::cout << "  here: ps_box:writefields" << std::endl;
    for (int i=0 ; i<psGroup.size(); i++ ) {
        psGroup[i].writeDensityField();
        psGroup[i].writeDensityFieldBinary();
        
        check_cudaError("writeFields in ps_box");
    }    
    for ( int i=0; i<potentials.size(); i++ ) {
        potentials[i]->writeBinaryOutput();
    }
    if ( verbose ) std::cout << "  Field written\n" << std::endl;
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

// Host endianness check; legacy VTK BINARY requires big-endian payloads.
static inline bool hostIsLittleEndian() {
    uint16_t x = 0x0001;
    return *reinterpret_cast<uint8_t*>(&x) == 0x01;
}

static inline float byteSwapFloat(float v) {
    uint32_t i;
    std::memcpy(&i, &v, 4);
    i = ((i >> 24) & 0x000000ffu) |
        ((i >>  8) & 0x0000ff00u) |
        ((i <<  8) & 0x00ff0000u) |
        ((i << 24) & 0xff000000u);
    std::memcpy(&v, &i, 4);
    return v;
}

// Write scalar field as a legacy VTK STRUCTURED_POINTS file (BINARY).
// Opens in ParaView directly and via pyvista.read() / vtk in Python.
void PS_Box::writeFieldVTK(const char* name, const float* dat) {
    FILE* otp = fopen(name, "wb");
    if ( otp == NULL ) { die("Failed to open output file in writeFieldVTK!"); }

    int nx = Nx[0];
    int ny = (Dim >= 2) ? Nx[1] : 1;
    int nz = (Dim >= 3) ? Nx[2] : 1;
    double sx = dx[0];
    double sy = (Dim >= 2) ? dx[1] : 1.0;
    double sz = (Dim >= 3) ? dx[2] : 1.0;

    fprintf(otp, "# vtk DataFile Version 3.0\n");
    fprintf(otp, "MATILDA.FT field\n");
    fprintf(otp, "BINARY\n");
    fprintf(otp, "DATASET STRUCTURED_POINTS\n");
    fprintf(otp, "DIMENSIONS %d %d %d\n", nx, ny, nz);
    fprintf(otp, "ORIGIN 0 0 0\n");
    fprintf(otp, "SPACING %g %g %g\n", sx, sy, sz);
    fprintf(otp, "POINT_DATA %d\n", M);
    fprintf(otp, "SCALARS field float 1\n");
    fprintf(otp, "LOOKUP_TABLE default\n");

    if ( hostIsLittleEndian() ) {
        float* buf = new float[M];
        for ( int i = 0; i < M; i++ ) buf[i] = byteSwapFloat(dat[i]);
        fwrite(buf, sizeof(float), M, otp);
        delete[] buf;
    } else {
        fwrite(dat, sizeof(float), M, otp);
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
    if ( otp == NULL ) { die("Failed to open output file in writeFieldFloat!"); }

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

    delete nn;
    delete r;
}

// write field of array vectors
void PS_Box::writeKFieldFloat(const char* name, const std::complex<float>* dat) {
    int i, j, * nn;
    nn = new int[Dim];
    FILE* otp;
    float* kv = new float [Dim];
    float k2;

    otp = fopen(name, "w");
    if ( otp == NULL ) { die("Failed to open output file in writeFieldFloat!"); }

    for (i = 0; i < M; i++) {
        k2 = get_kD(i, kv);
        unstack2(i, nn);

        for (j = 0; j < Dim; j++)
            fprintf(otp, "%f ", kv[j]);

        fprintf(otp, "%1.8e %1.8e %1.8e %1.8e\n", abs(dat[i]), sqrt(k2), real(dat[i]), imag(dat[i]));

        if (Dim == 2 && nn[0] == Nx[0] - 1)
            fprintf(otp, "\n");
    }

    fclose(otp);

    delete nn;
    delete kv;
}


// write field of array vectors
void PS_Box::writeKFieldFloat(const char* name, const float* dat) {
    int i, j, * nn;
    nn = new int[Dim];
    FILE* otp;
    float* kv = new float [Dim];
    float k2;

    otp = fopen(name, "w");
    if ( otp == NULL ) { die("Failed to open output file in writeFieldFloat!"); }

    for (i = 0; i < M; i++) {
        k2 = get_kD(i, kv);
        unstack2(i, nn);

        for (j = 0; j < Dim; j++)
            fprintf(otp, "%f ", kv[j]);

        fprintf(otp, "%1.8e %1.8e %1.8e\n", abs(dat[i]), sqrt(k2), dat[i] );

        if (Dim == 2 && nn[0] == Nx[0] - 1)
            fprintf(otp, "\n");
    }

    fclose(otp);

    delete nn;
    delete kv;
}




void PS_Box::writeTime() {

    int dt = time(0) - simTime;
    std::cout << "Total simulation time: " << dt / 60 << "m" << dt % 60 << "sec" << std::endl;
    
    dt = ftTimer;
    std::cout << "Total FT time: " << dt / 60 << "m" << dt % 60 << "sec" << std::endl;

    dt = ioTimer;
    std::cout << "Total I/O time: " << dt / 60 << "m" << dt % 60 << "sec" << std::endl;

}



PS_Box::~PS_Box() {
    cudaFree(d_thermoE);
    cudaFree(d_bondVirScratch);
    cudaFree(d_angleVirScratch);
    for ( int i=0 ; i<neighborLists.size(); i++ ) delete neighborLists[i];
}

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
    if ( id < 0 ) {
        std::string last_words = "Species label: " + testLabel + " not found!";
        die(last_words);
    }
    return id;
}

// Finds the index in the group vector with the label 'testLabel'
int PS_Box::findGroupInteger(std::string testLabel) {
    int id = -1;
    for ( int i=0 ; i<psGroup.size() ; i++ ) {
        // std::cout << "PSBox findGroup: " << testLabel << " " << psGroup[i].returnName() << " " << psGroup[i].isGroup(testLabel) << std::endl;
        if ( psGroup[i].isGroup( testLabel) ) {
            id = i;
            break;
        }
    }
    if ( id < 0 ) {
        std::string last_words = "PS_Box.cu: Group label not found: " + testLabel;
        die(last_words);
    }

    return id;
}


// Writes to LAMMPS .data file type
// Should be compatible with ``angle'' file type
void PS_Box::writeDataConfig(std::string filename) {
    
    std::ofstream out(filename);

    out << "Created by MATILDA.FT\n\n" ;

    out << nstot << " atoms" << "\n";
    out << nBondsTot << " bonds" << "\n";
    out << nAnglesTot << " angles" << "\n";

    // blank line
    out << "\n";

    // Find number of particle species
    int max = -342332;
    for ( int i=0 ; i<nstot ; i++ ) {
        if ( intSpecies[i] > max ) max = intSpecies[i];
    }
    out << max+1 << " atom types" << "\n";

    // Find number of bond types
    max = -1;
    for ( int i=0 ; i<nstot ; i++ ) {
        for ( int j=0 ; j<nBonds[i]; j++ ) 
            if ( bondType[i*MAXBONDS+j] > max ) max = bondType[i*MAXBONDS+j];
    }
    out << max+1 << " bond types" << "\n";

    // Find number of angle types
    max = -1;
    for ( int i=0 ; i<nstot ; i++ ) {
        for ( int j=0 ; j<nAngles[i]; j++ ) 
            if ( angleType[i*MAXANGLES+j] > max ) max = angleType[i*MAXBONDS+j];
    }
    out << max+1 << " angle types" << "\n";


    // blank line
    out << "\n";

    // Box dimensions
    out << "0.0 " << L[0] << " xlo xhi" << "\n";
    out << "0.0 " << L[1] << " ylo yhi" << "\n";
    if ( Dim > 2 ) out << "0.0 " << L[2] << " zlo zhi" << "\n";
    else out << "0.0 1.0 zlo zhi" << "\n";



    // blank line
    out << "\n";

    out << "Masses\n" << "\n";
    for ( int i=0 ; i<species.size(); i++ ) {
        out << i+1 << " " << species[i].mass << "\n";
    }


    // blank line
    out << "\n";


    // Write out particle coordinates and types
    out << "Atoms\n" << std::endl;
    for ( int i=0 ; i<nstot; i++ ) {

        out << i+1 << " " << mID[i]+1 << " " << intSpecies[i]+1 << "  " ;
        
        if ( this->doCharges ) {
            out << charges[i] << " " ;
        }

        for ( int j=0 ; j<Dim ; j++ ) 
            out << x[i*Dim+j] << " " ;
        if ( Dim < 3 )
            out << "0.0 " ;
        out << "\n";

    }// i=0 ; i<nstot


    // blank line
    out << "\n";


    // Write out bond list and types
    out << "Bonds\n" << "\n";
    int bondCounter = 0;
    for ( int i=0 ; i<nstot ; i++ ) {

        for ( int j=0 ; j<nBonds[i]; j++ ) {
            // Only write each bond once (when i < bondedTo)
            if ( i < bondedTo[i*MAXBONDS+j] ) {
                out << bondCounter+1 << " " << bondType[i*MAXBONDS+j]+1 << " " << i+1 << " " << bondedTo[i*MAXBONDS+j]+1 << "\n";
                
                bondCounter++;
            }
        }
    }

    if ( nAnglesTot > 0 ) {
        out << "\nAngles\n" << "\n";
        int angleCounter=0;
        for ( int i=0 ; i<nstot; i++ ) {

            for ( int j=0 ; j<nAngles[i]; j++ ) {
                // Only write bonds when i is first particle involved
                if ( i == angleGroup[3*(i*MAXANGLES+j)] ) {
                    out << angleCounter+1 << " " << angleType[i*MAXANGLES+j]+1 << " " << \
                        angleGroup[3*(i*MAXANGLES+j) + 0]+1 << " " << \
                        angleGroup[3*(i*MAXANGLES+j) + 1]+1 << " " << \
                        angleGroup[3*(i*MAXANGLES+j) + 2]+1 << "\n";
                        
                        angleCounter++;
                }
            }
        }
    }
    
    out.close();
}// end of writeDataConfig







void PS_Box::writeLammpsTraj(int step) {
    float *dtmp;
    dtmp = (float* ) malloc(nstot*Dim*sizeof(float));
    cudaMemcpy(dtmp, d_x, nstot*Dim*sizeof(float), cudaMemcpyDeviceToHost);
    for ( int i=0 ; i<nstot ; i++ ) {
        for ( int j=0 ; j<Dim ; j++ ) {
            x[i*Dim+j] = dtmp[i*Dim+j];
            if (x[i*Dim+j] > L[j]) x[i*Dim+j] -= L[j];
            else if ( x[i*Dim+j] < 0.0f ) x[i*Dim+j] += L[j];
        }
    }

    FILE* otp;
    int i, j;
    if (step == 0){
        otp = fopen(trajFileName.c_str(), "w");	
    }
    else 
        otp = fopen(trajFileName.c_str(), "a");

    fprintf(otp, "ITEM: TIMESTEP\n%d\nITEM: NUMBER OF ATOMS\n%d\n", step, nstot);
    fprintf(otp, "ITEM: BOX BOUNDS pp pp pp\n");
    fprintf(otp, "%f %f\n%f %f\n%f %f\n", 0.f, L[0], 0.f, L[1],
                (Dim == 3 ? 0.f : 0.f), (Dim == 3 ? L[2] : 1.f));

    // if ( Charges::do_charges )
    // 	fprintf(otp, "ITEM: ATOMS id type mol x y z q\n");
    // else
    fprintf(otp, "ITEM: ATOMS id type mol x y z\n");

    for (i = 0; i < nstot; i++) {
        fprintf(otp, "%d %d %d  ", i + 1, intSpecies[i] + 1, mID[i] + 1);
        for (j = 0; j < Dim; j++)
            fprintf(otp, "%f ", x[i*Dim+j]);

        for (j = Dim; j < 3; j++)
            fprintf(otp, "%f", 0.f);

        // if ( Charges::do_charges )
        // fprintf(otp, " %f", charges[i]);

        fprintf(otp, "\n");
    }
    fclose(otp);
    free(dtmp);
}

// Performs data reduction on device array d_dat
// Initially generated by Claude.ai
float PS_Box::sumDeviceArray(
    float *d_dat,   // [N] array to be summed
    int blockSize,  // blockSize for CUDA calls
    int N           // array size
    ) {

    float *d_output;
    int numBlocks = (N + blockSize - 1) / blockSize;
    float *h_output;// = new float[numBlocks];
    h_output = (float*) malloc(numBlocks * sizeof(float));

    // Allocate device memory
    cudaMalloc(&d_output, numBlocks * sizeof(float));
    
    // Launch kernel
    sumArrayKernel<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(
        d_dat, d_output, N);

   
    check_cudaError("sumDevArray");
    cudaDeviceSynchronize();

    // Copy partial results back to host
    cudaMemcpy(h_output, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Sum up partial results on CPU
    float totalSum = 0;
    for(int i = 0; i < numBlocks; i++) {
        totalSum += h_output[i];
    }
    
    // Cleanup
    cudaFree(d_output);
    free(h_output);

    check_cudaError("PS_Box::sumDevArray end");

    return totalSum;
}
void PS_Box::findSpinodal(std::istringstream& iss) {
    die("Invalid simulation type for a particle simulation!");
}

void PS_Box::modifyBox(std::istringstream& iss) {
    std::string word;

    iss >> word;

    if ( word == "potential" ) {
        iss >> word;

        if ( word == "remove" || word == "delete" ) {
            int potential_id;
            iss >> potential_id;

            if ( potential_id > 0 && potential_id <= potentials.size() ) {
                std::cout << "Removing potential " << potential_id << ", assuming potentials counted from 1" << std::endl;
                delete potentials[potential_id-1];
                potentials.erase( potentials.begin() + potential_id - 1 );
            }
        }
    }


    return;
}