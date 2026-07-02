// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "ps_potentialBias.h"
#include "PS_Box.h"

__global__ void d_extractCpxForceComp(cuComplex*, const cuComplex*, const int, const int, const int);


/*  Field Phase pairstyle
    The point of this type of interaction is to seed 
    specific fields and bias the system to form a target
    structure, such as lamellae, cylinders, etc. The idea 
    is to replace the chi-like interactions between species
    I  a static field w(r). 
	This will commonly be combined with a Helfand, 
    kappa-like self repulsions to help the system form
    a target phase. */


BiasField::BiasField(std::istringstream& iss, PS_Box* box) : PS_Potential(iss, box) {

    // Record the species acted upon
    iss >> grpI ;
    grpJ = grpI;

    // Prefactor for the bias
    iss >> Ao ;

    // String phase to bias towards
    std::string s1;
    iss >> s1;

    // Store phase name into known shorthand
    mybox->known_phase(s1, phase);


    // Default is one period
    n_periods = 1;

    // default direction is x direction
    dir = 0;

    
    // parse optional arguments
    while ( iss.tellg() != -1 ) {
        std::string temp_str;

        iss >> temp_str;
        if ( temp_str == "dir" ) {
            iss >> dir;
        }
        else if ( temp_str == "n_periods" ) {
            iss >> n_periods;
        }
        else {
            std::string err_msg = temp_str + " is not a valid initialize option in fts_potential";
            die(err_msg.c_str());
        }
    } // while (!iss)

    int M = mybox->M;
    int Dim = mybox->returnDimension();

    // allocate real-space force arrays
    cudaMalloc(&d_fr, M*Dim * sizeof(float));
    fr = (float*) malloc( M*Dim * sizeof(float));


    cudaMalloc(&d_fx, M * sizeof(float));
    fx = (float*) malloc( M * sizeof(float));

    if ( Dim >= 2 ) {
        cudaMalloc(&d_fy, M * sizeof(float));
        fy = (float*) malloc( M * sizeof(float));        
    }
    
    if ( Dim == 3 ) {
        cudaMalloc(&d_fz, M * sizeof(float));
        fz = (float*) malloc( M * sizeof(float));        
    }

    int is_ramping = ramp_check_input(iss, Ao);
}


void BiasField::initializePotential() {

    std::cout << "Initializing external bias field..." << std::endl;

    PS_Potential::initializePotential();


    std::complex<float> I(0.0, 1.0);
    float kv[3];
    int Dim = mybox->returnDimension();
    int M = mybox->M;
    int Grid = mybox->M_Grid;
    int Block = mybox->M_Block;

    // Define potential, force on host IN K-SPACE
    double *urDD = new double[M];
    mybox->make_bias_field(urDD, Ao, phase, dir, n_periods);
    
    for ( int i=0; i<M; i++ ) {
        ur[i] = urDD[i];
    }
    delete[] urDD;

    if ( mybox->verbose ) { 
        std::string bname = "ur-" + phase + ".dat";
        mybox->writeFieldFloat(bname.c_str(), ur); 
        bname = "ur-" + phase + ".vtk";
        mybox->writeFieldVTK(bname.c_str(), ur);
    }

    // Copy u(r) to device
    cudaMemcpy(this->d_ur, this->ur, M * sizeof(float), cudaMemcpyHostToDevice);


    // Place in complex array
    d_floatToCpx<<<Grid, Block>>>(mybox->d_cpxAlex, d_ur, M);

    // take d_cpxAlex = u(r) to k-space, place in uk
    mybox->cufftWrapperSingle(mybox->d_cpxAlex, d_uk, 1);

    // Copy FT(u(r)) back to host
    cudaMemcpy(uk, d_uk, M * sizeof(std::complex<float>), cudaMemcpyDeviceToHost);

    // Take spectral FT on host
    for (int i = 0; i < M; i++) {
        float k2 = mybox->get_kD(i, kv);
        float k  = sqrtf(k2);

        for (int j = 0; j < Dim; j++) {
            fk[i*Dim + j] = -I * kv[j] * uk[i];
        }
    }

    // Send k-space force to device
    cudaMemcpy(d_fk, fk, M * Dim * sizeof(std::complex<float>), cudaMemcpyHostToDevice);
    check_cudaError("NB_bias_potential: memory allocation");

    
    ///////////////////////////////////////////////////
    // Inverse FFT, store components in d_fx, fy, fz //
    ///////////////////////////////////////////////////

    // Extract x component
    d_extractCpxForceComp<<<Grid, Block>>>(mybox->d_cpxGabe, d_fk, 0, Dim, M);
    mybox->cufftWrapperSingle(mybox->d_cpxGabe, mybox->d_cpxAlex, -1);
    d_cpxToFloat<<<Grid, Block>>>(d_fx, mybox->d_cpxAlex, M);
    // Copy back to host
    cudaMemcpy(fx, d_fx, M*sizeof(float), cudaMemcpyDeviceToHost);
    
    if ( mybox->verbose ) { 
        std::string bname = "fx-" + phase + ".dat";
        mybox->writeFieldFloat(bname.c_str(), fx); 
        bname = "fx-" + phase + ".vtk";
        mybox->writeFieldVTK(bname.c_str(), fx);
    }

    // Extract y component
    if ( Dim >= 2 ) {
        d_extractCpxForceComp<<<Grid, Block>>>(mybox->d_cpxGabe, d_fk, 1, Dim, M);
        mybox->cufftWrapperSingle(mybox->d_cpxGabe, mybox->d_cpxAlex, -1);
        d_cpxToFloat<<<Grid, Block>>>(d_fy, mybox->d_cpxAlex, M);
        // Copy back to host
        cudaMemcpy(fy, d_fy, M*sizeof(float), cudaMemcpyDeviceToHost);
        
        if ( mybox->verbose ) { 
            std::string bname = "fy-" + phase + ".dat";
            mybox->writeFieldFloat(bname.c_str(), fy); 
            bname = "fy-" + phase + ".vtk";
            mybox->writeFieldVTK(bname.c_str(), fy);
        }
    }

    if ( Dim == 3 ) {
        d_extractCpxForceComp<<<Grid, Block>>>(mybox->d_cpxGabe, d_fk, 2, Dim, M);
        mybox->cufftWrapperSingle(mybox->d_cpxGabe, mybox->d_cpxAlex, -1);
        d_cpxToFloat<<<Grid, Block>>>(d_fz, mybox->d_cpxAlex, M);
        // Copy back to host
        cudaMemcpy(fz, d_fz, M*sizeof(float), cudaMemcpyDeviceToHost);        
        if ( mybox->verbose ) { 
            std::string bname = "fz-" + phase + ".dat";
            mybox->writeFieldFloat(bname.c_str(), fz); 
            bname = "fz-" + phase + ".vtk";
            mybox->writeFieldVTK(bname.c_str(), fz);
        }
    }

    // die("end of biasfield initiali potential");

}


// Accumulate the forces from the bias field on group Iind
void BiasField::CalcForces() {

    int Dim = mybox->returnDimension();

    // Accumulate the forces from the fields
    mybox->psGroup[Iind].accumulateGridForceComp(d_fx, 0);

    if ( Dim >= 2 ) {
        mybox->psGroup[Iind].accumulateGridForceComp(d_fy, 1);
    }

    if ( Dim == 3 ) {
        mybox->psGroup[Iind].accumulateGridForceComp(d_fz, 2);
    }

}


// Computes the energy of the density with the bias field as
// Ao * \int dr \, w(r) \, \rho_I(r)
float BiasField::CalcEnergy() {
    float *d_rhoI;
    int Grid = mybox->M_Grid;
    int Block = mybox->M_Block;
    int M = mybox->M;

    // Alex = rhoI * u(r)
    d_rhoI = mybox->psGroup[Iind].d_rho;
    d_multiplyFloatByFloat<<<Grid, Block>>>(mybox->d_Alex, d_rhoI, d_ur, M);

    energy = mybox->gvol * mybox->sumDeviceArray(mybox->d_Alex, Block, M);

    return energy;

}

BiasField::BiasField() {

}

BiasField::~BiasField() {

}



// phase = 0: Lamellar
// phase = 1: BCC
// phase = 2: CYL
// phase = 3: GYR
// __global__ void init_device_biasfield(
// 	float* ur, 
// 	float* fr,
// 	const int phase, 
// 	const float Ao, 
// 	const int dir,
// 	const int n_periods, 
// 	const float* dL, const float* dx,
// 	const int M, const int* Nx, const int Dim) 
// 	{


// 	const int ind = blockIdx.x * blockDim.x + threadIdx.x;
// 	if (ind >= M)
// 		return;

// 	float r[3];

// 	d_get_r(ind, r, Nx, dx, Dim);

// 	ur[ind] = 0.0f;

// 	// Initial lamellar fields, forces
// 	if (phase == 0) {
// 		float sin_arg = 2.0f * PI * float(n_periods) / dL[dir];
// 		ur[ind] = Ao * sin(sin_arg * r[dir]);

// 		for (int j = 0; j < Dim; j++) {
// 			if (j == dir)
// 				fr[ind * Dim + j] = Ao * sin_arg * cos(sin_arg * r[dir]);
// 			else
// 				fr[ind * Dim + j] = 0.f;
// 		}
// 	}

// 	// BCC phase
// 	// Assumes same number of periods in each direction
//     else if ( phase == 1 ) {

//         // Unit cell size
//         float a0 = dL[0] / float(n_periods);
        
//         float sr[3], dr[3], mdr2, dLh[3];
// 		for ( int j=0 ; j<Dim ; j++ ) {
// 			dLh[j] = 0.5 * dL[j];
// 		}

//         for ( int ix=0 ; ix<n_periods ; ix++ ) {
//             for ( int iy=0 ; iy<n_periods ; iy++ ) {
//                 for ( int iz=0 ; iz<n_periods ; iz++ ) {

// 					/////////////////////////////////////
//                     // Position of ``corner'' particle //
// 					/////////////////////////////////////
//                     sr[0] = ix * a0;
//                     sr[1] = iy * a0;
//                     sr[2] = iz * a0;

// 					// Distance from particle to current position
// 					mdr2 = d_pbc_mdr2(r, sr, dr, dL, dLh, Dim);

// 					float stdDev = 2.0;
// 					float variance = stdDev * stdDev;
// 					float expArg = -mdr2 / 2.0 / variance;

// 					// Gaussian potential with std dev of 2 hard-coded for now
// 					ur[ind] += -Ao * exp( expArg ); 

// 					fr[ind * Dim + 0] += -Ao * exp( expArg ) / variance * dr[0];
// 					fr[ind * Dim + 1] += -Ao * exp( expArg ) / variance * dr[1];
// 					fr[ind * Dim + 2] += -Ao * exp( expArg ) / variance * dr[2];



// 					////////////////////////////////////////////
//                     // Position of ``body-centered'' particle //
// 					////////////////////////////////////////////
//                     sr[0] = ix * a0 + 0.5 * a0;
//                     sr[1] = iy * a0 + 0.5 * a0;
//                     sr[2] = iz * a0 + 0.5 * a0;

// 					// Distance from particle to current position		
// 					mdr2 = d_pbc_mdr2(r, sr, dr, dL, dLh, Dim);

// 					// Gaussian potential with std dev of 2 hard-coded for now
// 					expArg = -mdr2 / 2.0 / variance;

// 					ur[ind] += -Ao * exp( expArg ); 

// 					fr[ind * Dim + 0] += -Ao * exp( expArg ) / variance * dr[0];
// 					fr[ind * Dim + 1] += -Ao * exp( expArg ) / variance * dr[1];
// 					fr[ind * Dim + 2] += -Ao * exp( expArg ) / variance * dr[2];

//                 }            
//             }            
//         }
//     }// if phase == 1

// 	// CYL phase
// 	// Assumes the same number of periods in both directions of the 
// 	// hexagonal plane of the cylinders. 
// 	// dir is interpretted as the direction of the cylinders
// 	// For 2D, this should be set to 2
// 	else if (phase == 2) {
// 		float dim1_arg, dim2_arg;
// 		int dim1 = 0, dim2 = 1;

// 		if (dir == dim1)
// 			dim1 = 2;
// 		else if (dir == dim2)
// 			dim2 = 2;

// 		dim1_arg = 2.0f * PI * float(n_periods) / dL[dim1];
// 		dim2_arg = 2.0f * PI * float(n_periods) / dL[dim2];

// 		ur[ind] = Ao * cos(dim1_arg * r[dim1]) * cos(dim2_arg * r[dim2]);

// 		if (Dim == 3)
// 			fr[ind * Dim + dir] = 0.0f;

// 		fr[ind * Dim + dim1] = Ao * dim1_arg * sin(dim1_arg * r[dim1]) * cos(dim2_arg * r[dim2]);
// 		fr[ind * Dim + dim2] = Ao * dim2_arg * sin(dim2_arg * r[dim2]) * cos(dim1_arg * r[dim1]);

// 	}

// 	// GYR phase
// 	// Assumes same number of periods in each direction
// 	else if (phase == 3) {
// 		if (Dim != 3) {
// 			return;
// 		}

// 		float args[3], cos_dir[3], sin_dir[3];
// 		for (int j = 0; j < Dim; j++) {
// 			args[j] = 2.0f * PI * float(n_periods) / dL[j];

// 			cos_dir[j] = cos(args[j] * r[j]);
// 			sin_dir[j] = sin(args[j] * r[j]);
// 		}

// 		float e_term = sin_dir[0] * cos_dir[1] + sin_dir[1] * cos_dir[2]
// 			+ sin_dir[2] * cos_dir[0];

// 		ur[ind] = Ao * (e_term * e_term - 1.44);

// 		fr[ind * Dim + 0] = -Ao * e_term * args[0] *
// 			(cos_dir[0] * cos_dir[1] - sin_dir[2] * sin_dir[0]);

// 		fr[ind * Dim + 1] = -Ao * e_term * args[1] *
// 			(cos_dir[1] * cos_dir[2] - sin_dir[0] * sin_dir[1]);

// 		fr[ind * Dim + 2] = -Ao * e_term * args[2] *
// 			(cos_dir[2] * cos_dir[0] - sin_dir[1] * sin_dir[2]);
// 	}

// }

// // Create the update function
// void BiasField::Update() {

//     if (!ramp) return;
// 	if (equil) return;
        

//     // The factor of 1000x is to attempt to cancel some float
//     // precision errors if chi is adjusted slowly over the course
//     // of a long simulation. It cancels since the ratio is passed
//     // as the argument

// 	double Ao = initial_prefactor + (final_prefactor-initial_prefactor)/double(prod_steps) * double(step+1);

// 	d_multiply_float_scalar<<<M_Grid, M_Block>>>(this->d_master_u, Ao, this->d_u, M);
// 	d_multiply_float_scalar<<<M_Grid * Dim, M_Block>>>(this->d_master_f, Ao, this->d_f, Dim*M);

// }


// void BiasField::ReportEnergies(int& die_flag){
// 	static int counter = 0;
// 	static string reported_energy = "";
// 	reported_energy += " " + to_string(energy) ;
// 	if (std::isnan(energy)) die_flag = 1 ;
    
//     if (++counter == num){
//         dout << reported_energy;
//         cout << "FieldPhasef: " + reported_energy;
//         counter=0;
// 		reported_energy.clear();
//     }
// }

// int    BiasField::num = 0;