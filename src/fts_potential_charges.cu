// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "include_libs.h"
#include "fts_potential_charges.h"
#include "FTS_Box.h"
#include "fts_species.h"
//#include "globals.h"

void die(const char*);
double ran2();
__global__ void d_accumulate_q_density(cuDoubleComplex*, const cuDoubleComplex*, const double, const int);

PotentialCharge::PotentialCharge(std::istringstream& iss, FTS_Box* p_box) : FTS_Potential(iss, p_box) {


    // Set stringstream to be ready to read kappa
    iss.seekg(0);
    std::string s1;
    iss >> s1;
    iss >> s1;

    potentialStyle = "Charges";

    iss >> E;
    iss >> delt;

    double ivalue = 0.0;
    wpl.resize(mybox->M,ivalue);
    d_wpl.resize(mybox->M, ivalue);
    d_Akpl.resize(mybox->M, ivalue);
    wplAlloc_flag = 1;

    d_rho_q.resize(mybox->M, ivalue);
    d_dHdw.resize(mybox->M, ivalue);
    
    if ( mybox->ftsStyle == "cl" ) {
        d_wNoise.resize(mybox->M, 0.0);
        std::cout << "Charge noise fields for CL allocated!" << std::endl;
    }
    
    // Set default update scheme
    updateScheme = "EM";

    while (iss.tellg() != -1 ) {
        iss >> s1;
        if ( s1 == "initialize" ) {
            initializeField(iss, wpl);
        }        
        

        else if ( s1 == "updateScheme" ) {
            iss >> updateScheme;
        }

        else if ( s1 == "modify" ) {
            iss >> s1;
            if ( s1 != "zeromean" && s1 != "zeroMean" ) { die("Invalid modify option on Charge potential"); }
            zeroMean = true;
        }
    }// optional arguments


    if ( updateScheme == "EMPC" ) {
        d_dHdwplo.resize(mybox->M, ivalue);        
        d_wplo.resize(mybox->M, ivalue);

        // ensure PC flag set to TRUE.
        mybox->PCflag = 1;
    }

    else if ( updateScheme == "1S" ) {
        die("1S scheme not implemented for charge potential");
    }

}// PotentialCharge constructor


// Updates potential fields using the chosen scheme
// If two-part predictor/corrector scheme is to be used, then 
// this step is the predictor step
void PotentialCharge::updateFields() {
    
    bool doCL = false;
    if ( mybox->ftsStyle == "cl" ) doCL = true;


    // cast thrust vectors to cuDoubleComplex for use in kernel
    cuDoubleComplex* _d_dHdw =   (cuDoubleComplex*)thrust::raw_pointer_cast(d_dHdw.data());
    cuDoubleComplex* _d_wpl =    (cuDoubleComplex*)thrust::raw_pointer_cast(d_wpl.data());
    cuDoubleComplex* _d_rho_q =  (cuDoubleComplex*)thrust::raw_pointer_cast(d_rho_q.data());
    cuDoubleComplex* _d_wNoise = (cuDoubleComplex*)thrust::raw_pointer_cast(d_wNoise.data());


    // Initialize to zero
    thrust::fill(d_rho_q.begin(), d_rho_q.end(), 0.0);

    
    // Loop over species, adding them to the field
    for ( int i=0 ; i<mybox->Species.size() ; i++ ) {
        if ( fabs(mybox->Species[i].charge) > 1.0E-8 ) {

            cuDoubleComplex* _d_rho_tp = (cuDoubleComplex*)thrust::raw_pointer_cast(mybox->Species[i].d_density.data());
            int grid = mybox->M_Grid;
            int block = mybox->M_Block;
            int M = mybox->M;

            d_accumulate_q_density<<<grid, block>>>(_d_rho_q, _d_rho_tp, mybox->Species[i].charge, M);

        }
    }


    // Generate noise fields if doing CL
    if ( doCL ) {
        double noiseMag = sqrt(2.0 * delt / mybox->gvol );
        d_makeDoubleNoise<<<mybox->M_Grid, mybox->M_Block>>>(_d_wNoise, mybox->d_states, noiseMag, mybox->M);
    }


    // Get square gradient of field wpl, store in cpxAtmn
    // d_cpxAtmn = \nabla^2 _d_wpl
    mybox->computeGrad2FieldDouble(mybox->d_cpxAtmn, _d_wpl, 1);

    // Make the force in real space
    d_makeChargeForce<<<mybox->M_Grid, mybox->M_Block>>>(_d_dHdw, mybox->d_cpxAtmn, _d_rho_q,
        E, mybox->Nr, mybox->M);


    if ( updateScheme == "EMPC" ) {
        storePredictorData();
    }


    // Update the fields
    if ( updateScheme == "EM" || updateScheme == "EMPC") {
        d_fts_updateEM<<<mybox->M_Grid, mybox->M_Block>>>(_d_wpl, _d_dHdw, _d_wNoise, doCL, delt, mybox->M);
    }


    else if ( updateScheme == "1S" ) {
        die("1S not set up for fts_potential_charges");
        // // Put the force and potential into k-space
        // mybox->cufftWrapperDouble(d_dHdw, d_dHdw, 1);
        // mybox->cufftWrapperDouble(d_wpl, d_wpl, 1);

        // // Pointer to linear coefficient
        // cuDoubleComplex* _d_Ak = (cuDoubleComplex*)thrust::raw_pointer_cast(d_Akpl.data());

        // // Call updater
        // d_fts_update1S<<<mybox->M_Grid, mybox->M_Block>>>(_d_wpl, _d_dHdw, _d_Ak, delt, mybox->M);

        // // Bring potential back to r-space
        // mybox->cufftWrapperDouble(d_wpl, d_wpl, -1);
    }

    // Check for modifiers
    if ( zeroMean == true ) {
        thrust::complex<double> mean = thrust::reduce(d_wpl.begin(), d_wpl.end()) / double(mybox->M);
        
        // dtmp = mean
        thrust::device_vector<thrust::complex<double>> dtmp(mybox->M, mean);

        // wpl(r) = wpl(r) - mean
        thrust::transform(d_wpl.begin(), d_wpl.end(), dtmp.begin(), d_wpl.begin(), 
            thrust::minus<thrust::complex<double>>());
    }

}// updateFields




void PotentialCharge::correctFields() {
    if ( updateScheme != "EMPC" ) {
        return;
    }

    bool doCL = false;
    if ( mybox->ftsStyle == "cl" ) doCL = true;


    // cast thrust vectors to cuDoubleComplex for use in kernel
    cuDoubleComplex* _d_dHdw =   (cuDoubleComplex*)thrust::raw_pointer_cast(d_dHdw.data());
    cuDoubleComplex* _d_dHdwplo =   (cuDoubleComplex*)thrust::raw_pointer_cast(d_dHdwplo.data());
    cuDoubleComplex* _d_wpl =    (cuDoubleComplex*)thrust::raw_pointer_cast(d_wpl.data());
    cuDoubleComplex* _d_wplo =    (cuDoubleComplex*)thrust::raw_pointer_cast(d_wplo.data());

    cuDoubleComplex* _d_rho_q =  (cuDoubleComplex*)thrust::raw_pointer_cast(d_rho_q.data());
    cuDoubleComplex* _d_wNoise = (cuDoubleComplex*)thrust::raw_pointer_cast(d_wNoise.data());


    // Initialize to zero
    thrust::fill(d_rho_q.begin(), d_rho_q.end(), 0.0);

    
    // Loop over species, adding them to the field
    for ( int i=0 ; i<mybox->Species.size() ; i++ ) {
        if ( fabs(mybox->Species[i].charge) > 1.0E-8 ) {

            cuDoubleComplex* _d_rho_tp = (cuDoubleComplex*)thrust::raw_pointer_cast(mybox->Species[i].d_density.data());
            int grid = mybox->M_Grid;
            int block = mybox->M_Block;
            int M = mybox->M;

            d_accumulate_q_density<<<grid, block>>>(_d_rho_q, _d_rho_tp, mybox->Species[i].charge, M);

        }
    }


    // Get square gradient of field wpl, store in cpxAtmn
    // d_cpxAtmn = \nabla^2 _d_wpl
    mybox->computeGrad2FieldDouble(mybox->d_cpxAtmn, _d_wpl, 1);

    // Make the force in real space
    d_makeChargeForce<<<mybox->M_Grid, mybox->M_Block>>>(_d_dHdw, mybox->d_cpxAtmn, _d_rho_q,
        E, mybox->Nr, mybox->M);


    // Corrector step for field updates
    d_fts_updateEMPC<<<mybox->M_Grid, mybox->M_Block>>>(_d_wpl, _d_wplo, _d_dHdw, _d_dHdwplo, _d_wNoise, doCL, delt, mybox->M);
    

    
    // Check for modifiers
    if ( zeroMean == true ) {
        thrust::complex<double> mean = thrust::reduce(d_wpl.begin(), d_wpl.end()) / double(mybox->M);
        
        // dtmp = mean
        thrust::device_vector<thrust::complex<double>> dtmp(mybox->M, mean);

        // wpl(r) = wpl(r) - mean
        thrust::transform(d_wpl.begin(), d_wpl.end(), dtmp.begin(), d_wpl.begin(), 
            thrust::minus<thrust::complex<double>>());
    }    
}



// Stores initial potential and force forms for predictor/corrector updates
void PotentialCharge::storePredictorData() {
    d_wplo = d_wpl;
    d_dHdwplo = d_dHdw;
}




// This routine computes the ``force'' on the field conjugate to charge interactions
__global__ void d_makeChargeForce(
    cuDoubleComplex* dHdw,              // Field holding dHdw
    const cuDoubleComplex* grad2w,      // \nabla^2 d_wpl
    const cuDoubleComplex* rho_q,       // [M] Charge density field
    const double E,                     // Dimensionless Bjerrum length
    const double Nr,                    // Reference chain length
    const int M                         // number of grid points
    ) {

    const int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= M)
        return;

    dHdw[ind].x = -grad2w[ind].x / E - rho_q[ind].y / Nr;
    dHdw[ind].y = -grad2w[ind].y / E + rho_q[ind].x / Nr;
}

__global__ void d_accumulate_q_density(
    cuDoubleComplex* rho_q,             // [M] storage for total charge
    const cuDoubleComplex* species_rho, // [M] species density
    const double q,                     // magnitude of charge
    const int M                         // array size
    ) {

    const int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= M)
        return;

    rho_q[ind].x += species_rho[ind].x * q;
    rho_q[ind].y += species_rho[ind].y * q;

}


void PotentialCharge::writeFields(int potInd ) { 
    char nm[30];
    sprintf(nm, "wpl_Charge%d.dat", potInd);

    // Transfer field to the host;
    wpl = d_wpl;
    mybox->writeTComplexGridData(nm, wpl);
}


// Computes this potential's contribution to the effective Hamiltonian
std::complex<double> PotentialCharge::calcHamiltonian() {
    thrust::device_vector<thrust::complex<double>> dtmp(mybox->M);

    cuDoubleComplex* _d_wpl = (cuDoubleComplex*)thrust::raw_pointer_cast(d_wpl.data());

    // d_cpxAtmn = \nabla^2 wpl
    mybox->computeGrad2FieldDouble(mybox->d_cpxAtmn, _d_wpl, 1);

    thrust::device_ptr<thrust::complex<double>> d_grad2w((thrust::complex<double>*)mybox->d_cpxAtmn);

    // dtmp(r) = wpl(r) * grad2wpl(r)
    thrust::transform(d_wpl.begin(), d_wpl.end(), d_grad2w, dtmp.begin(),
        thrust::multiplies<thrust::complex<double>>());

    thrust::complex<double> integral = thrust::reduce(dtmp.begin(), dtmp.end()) * mybox->gvol;

    // H = (1/2E) integral |grad(wpl)|^2 dr = -(1/2E) integral wpl * grad2(wpl) dr
    Hterm = -integral / 2.0 / E;

    // std::cout << Hterm << " " << E << " " << integral << " " << mybox->gvol << std::endl;

    // wpl = dtmp;
    // mybox->writeTComplexGridData("grad2_phi.dat", wpl);

    // wpl = d_wpl;
    // mybox->writeTComplexGridData("phi.dat", d_wpl);

    return Hterm;

}

void PotentialCharge::initLinearCoeffs() {
    // Akpl = C / kappaN
    thrust::fill(d_Akpl.begin(), d_Akpl.end(), 1.0/E);
}

PotentialCharge::~PotentialCharge() {}