// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "ps_potentialBias.h"
#include "PS_Box.h"



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

    // Prefactor for the bias
    iss >> Ao ;

    // String phase to bias towards
    iss >> phase;

    // To add: BCC
    // This list of phases should go in the Box.cu routine at some point
    // and checked against that global list. Make it a string vector like
    // the quotes database
    if ( phase != "L" && phase != "LAM" && 
         phase != "G" && phase != "GYR" && 
         phase != "BCC" && phase != "SPH" && phase != "S" &&
         phase != "C" && phase != "CYL" && phase != "H" && phase != "HEX" ) {
            die("Invalid phase provided to potential bias!");
         }

    // Save the number of periods
    iss >> n_periods;

    // default direction is x direction
    dir = 0;

    std::string temp_str;
    
    if ( iss.tellg() == -1 ) {
        iss >> temp_str;

        if ( temp_str == "dir" || temp_str == "direction" ) {
            iss >> dir;
        }
    }

}


void BiasField::initializePotential() {

    std::cout << "Initializing external bias field..." << std::endl;

    PS_Potential::initializePotential();


    std::complex<float> I(0.0, 1.0);
    float kv[3], k2;
    int Dim = mybox->returnDimension();
    int M = mybox->M;

    // printf("Setting up FieldPhase pair style..."); fflush(stdout);

	// if ( phase == 1 && Dim != 3 ) {
	// 	die("BCC bias only compatible with 3D simulations!");
	// }

    // init_device_biasfield<<<M_Grid, M_Block>>>(this->d_u, this->d_f,
    //     phase, initial_prefactor, dir, n_periods, d_L, d_dx, M, d_Nx, Dim);

    // init_device_biasfield<<<M_Grid, M_Block>>>(this->d_master_u, this->d_master_f,
    //     phase, 1.0f, dir, n_periods, d_L, d_dx, M, d_Nx, Dim);

    // cudaMemcpy(this->u, this->d_u, M * sizeof(float), cudaMemcpyDeviceToHost);
    // cudaMemcpy(this->u, this->d_u, M * sizeof(float), cudaMemcpyDeviceToHost); 

    // write_grid_data("biasfield_map.dat", this->u);

    // printf("done!\n"); fflush(stdout);
}

void BiasField::CalcForces() {


    // // X-component of the force
    // d_prepareFieldForce<<<M_Grid, M_Block>>>(d_cpx1, d_cpx2,
    //     d_all_rho, this->d_f, type1, type2, 0, Dim, M);
    // d_accumulateGridForce<<<M_Grid, M_Block>>>(d_cpx1,
    //     d_all_rho, d_all_fx, type1, M);
    // d_accumulateGridForce<<<M_Grid, M_Block>>>(d_cpx2,
    //     d_all_rho, d_all_fx, type2, M);




    // // Y-component of the force
    // d_prepareFieldForce<<<M_Grid, M_Block>>>(d_cpx1, d_cpx2,
    //     d_all_rho, this->d_f, type1, type2, 1, Dim, M);
    // d_accumulateGridForce<<<M_Grid, M_Block>>>(d_cpx1,
    //     d_all_rho, d_all_fy, type1, M);
    // d_accumulateGridForce<<<M_Grid, M_Block>>>(d_cpx2,
    //     d_all_rho, d_all_fy, type2, M);


    // if (Dim == 3) {
    //     // Z-component of the force
    //     d_prepareFieldForce<<<M_Grid, M_Block>>>(d_cpx1, d_cpx2,
    //         d_all_rho, this->d_f, type1, type2, 2, Dim, M);
    //     d_accumulateGridForce<<<M_Grid, M_Block>>>(d_cpx1,
    //         d_all_rho, d_all_fz, type1, M);
    //     d_accumulateGridForce<<<M_Grid, M_Block>>>(d_cpx2,
    //         d_all_rho, d_all_fz, type2, M);
    // }


}

float BiasField::CalcEnergy() {

	return -67.0f;
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