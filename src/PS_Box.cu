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

Integrator* IntegratorFactory(std::istringstream&, PS_Box*);
__global__ void d_testIntegrator(float*, const int, const int, curandState*);

// Executes the commands for a given time step
// Updates all fields, then recomputes all molecule densities
// then populating species densities
void PS_Box::doTimeStep(int step) {


    // First integration step, when needed (e.g., velo Verlet)
    for ( int i=0 ; i<integrators.size(); i++ ) {
        // std::cout << "Integrate 1..." ; fflush(stdout);
        integrators[i]->Integrate_1();
        // std::cout << "done!" << std::endl;
    }



    // Update grid weights
    // d_calcGridWeights<<<nsGrid, nsBlock>>>(_d_gridW, _d_gridInds, d_x, _d_Nx, 
    //     d_dxf, nstot, pmeorder, M, Dim );
    // check_cudaError("Weights calculated in PS_Box");

    ///////////////////////////
    // UPDATE DENSITY FIELDS //
    ///////////////////////////

    // update the density fields
    // for ( int i=0 ; i<psGroup.size(); i++ ) {
    //     // zero density, grid force fields
    //     psGroup[i].zeroFields();

    //     // Fill density field
    //     psGroup[i].makeDensityField();
    // }

    // Zero particle forces
    // std::cout << "zero forces..." ; fflush(stdout);
    d_assignFloatVal<<<DnsGrid, nsBlock>>>(_d_f, 0.0, Dim*nstot);
    // std::cout << "done!" << std::endl;


    ////////////////////
    // COMPUTE FORCES //
    ////////////////////
    forces();


    // Second integration step
    for ( int i=0 ; i<integrators.size(); i++ ) {
        // std::cout << "integrate 2..." ; fflush(stdout);
        integrators[i]->Integrate_2();
        // std::cout << "done!" << std::endl;
    }


    ///////////////
    // I/O steps //
    ///////////////

    // Write log data
    if ( step % logFreq == 0 ) {
        // std::cout << "log..." ; fflush(stdout);
        writeData(step);
        // std::cout << "done!" << std::endl;
    }

    // write to GSD traj file
    if ( gsdFreq > 0 && step % gsdFreq == 0 ) { 
        // std::cout << "gsd..." ; fflush(stdout);
        writeGSDtraj();
        // std::cout << "done!" << std::endl;
    }

    if ( trajFreq > 0 && step % trajFreq == 0 ) {
        // std::cout << "lammpstrj..." ; fflush(stdout);
        writeLammpsTraj(step);
        // std::cout << "done!" << std::endl;
    }

    // Write field data
    // if ( fieldFreq > 0 && step % fieldFreq == 0 ) {
    //     writeFields();
    // }

} // doTimeStep


void PS_Box::NVT(int maxSteps) {
    std::cout << "RUNNING NVT?!?" << std::endl;
    
    for ( int i=0 ; i<maxSteps; i++ ) {
        doTimeStep(i);

        totSteps++;
    }
    std::cout << "NVT Finished, writing final output" << std::endl;
    writeData(maxSteps);
    // writeFields();
    writeGSDtraj(); 

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
    std::cout << "step: " << step ;

    OTP << std::endl;
    std::cout << std::endl;

    OTP.close();
}




// Currently loops over groups and writes field-based densities
void PS_Box::writeFields() {
    std::cout << "  here: ps_box:writefields" << std::endl;
    for (int i=0 ; i<psGroup.size(); i++ ) {
        psGroup[i].writeDensityField();
        
        check_cudaError("writeFields in ps_box");
    }    
    std::cout << "  Field written\n" << std::endl;
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




// Sends all particle-size arrays from host to device. Intended to be used after
// initialization when info needs to go to device for running simulations, though
// could be used any time.
void PS_Box::sendAllHostToDevice(void) {
    
    float *xtmp;
    xtmp = (float*) calloc( nstot*Dim, sizeof(float));

    for ( int i=0; i<nstot*Dim; i++ ) {
        xtmp[i] = x[i];
    }

    //d_x = x;
    cudaMemcpy(d_x, xtmp, nstot*Dim * sizeof(float), cudaMemcpyHostToDevice);
    
    // d_v = v;
    // d_f = f;

    float dxf[3];
    if ( Dim > 3 ) die("Dim greater than 3?!??");
    for ( int j=0 ; j<Dim ; j++ ) {
        dxf[j] = (float)dx[j];
    }
    cudaMemcpy(d_dxf, dxf, Dim*sizeof(float), cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_L, L, Dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Lh, Lh, Dim*sizeof(float), cudaMemcpyHostToDevice);
    
    d_intSpecies = intSpecies;
    
    // d_nBonds = nBonds;
    // d_bondedTo = bondedTo;
    // d_bondType = bondType;

    // d_bondStyle = bondStyle;
    // d_bondK = bondK;
    // d_bondReq = bondReq;

    // d_nAngles = nAngles;
    // d_angleGroup = angleGroup;
    // d_angleType = angleType;

    // d_angleTheq = angleTheq;
    // d_angleK = angleK;
    // d_angleStyle = angleStyle;

    free(xtmp);

}






__global__ void d_testIntegrator(
	float* x, 				// [nstot*Dim] positiosn to be updated
	const int ns, 			// number of sites in group to be integrated (ns <= nstot)
	const int Dim, 			// system dimensionality
	curandState* d_states	// [nstot*Dim] RNG state variable
	) {

	int ind = blockIdx.x * blockDim.x + threadIdx.x;
	if (ind >= ns)
		return;
		
    curandState l_state;
    l_state = d_states[ind];

    x[ind*Dim+0] = 1.55 + curand_normal(&l_state) * 0.01;
    x[ind*Dim+1] = 6.966 + curand_normal(&l_state) * 2.0;

	d_states[ind] = l_state;
}