// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "include_libs.h"
#include "fts_potential_incompress.h"
#include "FTS_Box.h"
#include "fts_species.h"
//#include "globals.h"

void die(const char*);
double ran2();


PotentialIncompress::PotentialIncompress(std::istringstream& iss, FTS_Box* p_box) : FTS_Potential(iss, p_box) {


    // Set stringstream to be ready to read kappa
    iss.seekg(0);
    std::string s1;
    iss >> s1;
    iss >> s1;

    potentialStyle = "Incompress";

    iss >> delt;

    double ivalue = 0.0;
    wpl.resize(mybox->M,ivalue);
    d_wpl.resize(mybox->M, ivalue);
    d_Akpl.resize(mybox->M, ivalue);
    wplAlloc_flag = 1;

    d_rho_total.resize(mybox->M, ivalue);
    d_dHdw.resize(mybox->M, ivalue);

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
            if ( s1 != "zeromean" && s1 != "zeroMean" ) { die("Invalid modify option on Helfand potential"); }
            zeroMean = true;
        }
    }// optional arguments

    d_wpl = wpl;

    if ( updateScheme == "EMPC" ) {
        d_dHdwplo.resize(mybox->M, ivalue);        
        d_wplo.resize(mybox->M, ivalue);

        // ensure PC flag set to TRUE.
        mybox->PCflag = 1;
    }

}// PotentialHelfand constructor


// Updates potential fields using the chosen scheme
// If two-part predictor/corrector scheme is to be used, then 
// this step is the predictor step
void PotentialIncompress::updateFields() {

    // Initialize to zero
    thrust::fill(d_rho_total.begin(), d_rho_total.end(), 0.0);

    
    // Loop over species, adding them to the field
    for ( int i=0 ; i<mybox->Species.size() ; i++ ) {
        thrust::transform(mybox->Species[i].d_density.begin(), mybox->Species[i].d_density.begin()+mybox->M,
            d_rho_total.begin(), d_rho_total.begin(), thrust::plus<thrust::complex<double>>());
    }



    // cast thrust vectors to cuDoubleComplex for use in kernel
    cuDoubleComplex* _d_dHdw = (cuDoubleComplex*)thrust::raw_pointer_cast(d_dHdw.data());
    cuDoubleComplex* _d_wpl = (cuDoubleComplex*)thrust::raw_pointer_cast(d_wpl.data());
    cuDoubleComplex* _d_rho_total = (cuDoubleComplex*)thrust::raw_pointer_cast(d_rho_total.data());

    // Make the force in real space
    d_makeIncompressForce<<<mybox->M_Grid, mybox->M_Block>>>(_d_dHdw, _d_wpl, _d_rho_total, mybox->C,
        mybox->Nr, mybox->M);



    if ( updateScheme == "EMPC" ) {
        storePredictorData();
    }


    // Update the fields
    if ( updateScheme == "EM" || updateScheme == "EMPC") {
        d_fts_updateEM<<<mybox->M_Grid, mybox->M_Block>>>(_d_wpl, _d_dHdw, nullptr, false, delt, mybox->M);
    }


    else if ( updateScheme == "1S" ) {
        // Put the force and potential into k-space
        mybox->cufftWrapperDouble(d_dHdw, d_dHdw, 1);
        mybox->cufftWrapperDouble(d_wpl, d_wpl, 1);

        // Pointer to linear coefficient
        cuDoubleComplex* _d_Ak = (cuDoubleComplex*)thrust::raw_pointer_cast(d_Akpl.data());

        // Call updater
        d_fts_update1S<<<mybox->M_Grid, mybox->M_Block>>>(_d_wpl, _d_dHdw, _d_Ak, delt, mybox->M);

        // Bring potential back to r-space
        mybox->cufftWrapperDouble(d_wpl, d_wpl, -1);
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




void PotentialIncompress::correctFields() {
    if ( updateScheme != "EMPC" ) {
        return;
    }

    // Initialize to zero
    thrust::fill(d_rho_total.begin(), d_rho_total.end(), 0.0);

    
    // Loop over species, adding them to the field
    for ( int i=0 ; i<mybox->Species.size() ; i++ ) {
        thrust::transform(mybox->Species[i].d_density.begin(), mybox->Species[i].d_density.begin()+mybox->M,
            d_rho_total.begin(), d_rho_total.begin(), thrust::plus<thrust::complex<double>>());
    }



    // cast thrust vectors to cuDoubleComplex for use in kernel
    cuDoubleComplex* _d_dHdw = (cuDoubleComplex*)thrust::raw_pointer_cast(d_dHdw.data());
    cuDoubleComplex* _d_wpl = (cuDoubleComplex*)thrust::raw_pointer_cast(d_wpl.data());
    cuDoubleComplex* _d_rho_total = (cuDoubleComplex*)thrust::raw_pointer_cast(d_rho_total.data());

    cuDoubleComplex* _d_dHdwplo = (cuDoubleComplex*)thrust::raw_pointer_cast(d_dHdwplo.data());
    cuDoubleComplex* _d_wplo = (cuDoubleComplex*)thrust::raw_pointer_cast(d_wplo.data());

    // Make the force in real space
    d_makeIncompressForce<<<mybox->M_Grid, mybox->M_Block>>>(_d_dHdw, _d_wpl, _d_rho_total, mybox->C,
        mybox->Nr, mybox->M);


    // Corrector step for field updates
    d_fts_updateEMPC<<<mybox->M_Grid, mybox->M_Block>>>(_d_wpl, _d_wplo, _d_dHdw, _d_dHdwplo, nullptr, false, delt, mybox->M);
    

    
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
void PotentialIncompress::storePredictorData() {
    d_wplo = d_wpl;
    d_dHdwplo = d_dHdw;
}




// This routine is currently written to deal with dHdw in real space
// the I*rho0 term should change if it changed to k-space updating
__global__ void d_makeIncompressForce(
    cuDoubleComplex* dHdw,              // Field holding dHdw
    const cuDoubleComplex* w,           // current d_wpl
    const cuDoubleComplex* rho_total,   // current total density
    const double C,                     // Polymer concentration, based on Nr
    const double Nr,                    // Reference chain length
    const int M                         // number of grid points
    ) {

    const int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= M)
        return;

    // dHdw = C * w / kN - I * C + I * rho_total / Nr ;
    dHdw[ind].x = -rho_total[ind].y / Nr;
    dHdw[ind].y = rho_total[ind].x / Nr - C;
}


void PotentialIncompress::writeFields(int potInd ) { 
    char nm[30];
    sprintf(nm, "wpl_Incompress%d.dat", potInd);

    // Transfer field to the host;
    wpl = d_wpl;
    mybox->writeTComplexGridData(nm, wpl);
}


// Computes this potential's contribution to the effective Hamiltonian
std::complex<double> PotentialIncompress::calcHamiltonian() {
    thrust::device_vector<thrust::complex<double>> dtmp(mybox->M);
    thrust::complex<double> I(0.0,1.0);



    // -i C * int(wpl)
    thrust::complex<double> integral = thrust::reduce(d_wpl.begin(), d_wpl.end()) * mybox->gvol;

    Hterm = -I * mybox->C * integral;

    //std::cout << Hterm << std::endl;

    return Hterm;
    
}

void PotentialIncompress::initLinearCoeffs() {
    
    thrust::fill(d_Akpl.begin(), d_Akpl.end(), 0.0);
}

PotentialIncompress::~PotentialIncompress() {}