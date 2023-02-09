// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "include_libs.h"
#include "fts_potential_flory.h"
#include "FTS_Box.h"
#include "fts_species.h"


void die(const char*);
double ran2();


// Note: this Flory code will ALWAYS use the multispecies method of 
// Koski, Chao, Riggleman JCP 2013 even if there are only two species
// This creates a potentially redundant field, but makes coding the
// general case simpler. In our experience, initial field conditions
// only really "matter" on the wmi field, so al initialize commands
// apply only to the real part of that field.
PotentialFlory::PotentialFlory(std::istringstream& iss, FTS_Box* p_box) : FTS_Potential(iss, p_box) {
    
    // Set stringstream to be ready to read kappa
    iss.seekg(0);
    std::string s1;
    iss >> s1;
    iss >> s1;

    potentialStyle = "Flory";

    iss >> typeI;
    iss >> typeJ;
    std::cout << "Flory potential acting on species " << typeI << ", " << typeJ << std::endl;

    actsOn.push_back(typeI);
    actsOn.push_back(typeJ);

    iss >> chiN;

    // Set time steps for plus and minus fields
    iss >> deltPlus;
    iss >> deltMinus;
    

    // Resize all of the (+) and (-) arrays
    double ivalue = 0.0;
    wpl.resize(mybox->M,ivalue);
    d_wpl.resize(mybox->M, ivalue);
    d_Akpl.resize(mybox->M, ivalue);

    wmi.resize(mybox->M,ivalue);
    d_wmi.resize(mybox->M, ivalue);
    d_Akmi.resize(mybox->M, ivalue);
    
    
    updateScheme = "EM";

    while (iss.tellg() != -1 ) {
        iss >> s1;
        if ( s1 == "initialize" ) {
            iss >> s1;
            if ( s1 == "value" ) {
                double rVal, iVal;
                iss >> rVal;
                iss >> iVal;
                thrust::fill(wmi.begin(), wmi.end(), std::complex<double>(rVal, iVal));
            }
            // Two floats expected: amplitude of noise on real part and imag part
            else if ( s1 == "random" ) {
                double rAmp, iAmp;
                iss >> rAmp;
                iss >> iAmp;
                // Fill host field with random noise
                for ( int i=0 ; i<mybox->M ; i++ ) {
                    wmi[i] = std::complex<double>(rAmp * ran2(), iAmp * ran2() );
                }
                
            }
            
            // Expects an int and two doubles [int dir] [double amplitude] [double period]
            else if ( s1 == "sin" || s1 == "sine" ) {
                double amp, period;
                int dir;
                iss >> dir;
                iss >> amp;
                iss >> period; 

                for ( int i=0 ; i<mybox->M ; i++ ) {
                    double r[3];
                    mybox->get_r(i, r);
                    wmi[i] = amp * sin(2.0 * PI * r[dir] * period / mybox->L[dir]);
                }

            }

            else {
                die("Invalid initialize option on potential Flory");
            }
        }

        else if ( s1 == "updateScheme" ) {
            iss >> updateScheme;
        }

        else if ( s1 == "modify" ) {
            iss >> s1;
            if ( s1 != "zeromean" && s1 != "zeroMean" ) { die("Invalid modify option on Flory potential"); }
            zeroMean = true;
        }
    }// optional arguments


    // transfer to device
    d_wpl = wpl;
    d_wmi = wmi;
    
}


void PotentialFlory::updateFields() {

    thrust::device_vector<thrust::complex<double>> d_rhoI(mybox->M);
    thrust::device_vector<thrust::complex<double>> d_rhoJ(mybox->M);
    thrust::device_vector<thrust::complex<double>> d_dHdwpl(mybox->M);
    thrust::device_vector<thrust::complex<double>> d_dHdwmi(mybox->M);

    // Assign species I and J
    for ( int i=0 ; i<mybox->Species.size() ; i++ ) {
        if ( actsOn[0] == mybox->Species[i].fts_species ) { d_rhoI = mybox->Species[i].d_density; }
        else if ( actsOn[1] == mybox->Species[i].fts_species ) { d_rhoJ = mybox->Species[i].d_density; }
    }

    // Pointers to the thrust data
    cuDoubleComplex* _d_dHdwpl = (cuDoubleComplex*)thrust::raw_pointer_cast(d_dHdwpl.data());
    cuDoubleComplex* _d_dHdwmi = (cuDoubleComplex*)thrust::raw_pointer_cast(d_dHdwmi.data());
    cuDoubleComplex* _d_rhoI = (cuDoubleComplex*)thrust::raw_pointer_cast(d_rhoI.data());
    cuDoubleComplex* _d_rhoJ = (cuDoubleComplex*)thrust::raw_pointer_cast(d_rhoJ.data());
    cuDoubleComplex* _d_wpl = (cuDoubleComplex*)thrust::raw_pointer_cast(d_wpl.data());
    cuDoubleComplex* _d_wmi = (cuDoubleComplex*)thrust::raw_pointer_cast(d_wmi.data());

    // Forces are generated in real space
    d_makeFloryForce<<<mybox->M_Grid, mybox->M_Block>>>(_d_dHdwpl, _d_dHdwmi, _d_wpl,
        _d_wmi, _d_rhoI, _d_rhoJ, mybox->C, chiN, mybox->Nr, mybox->M);
        
    // debug stuff
    mybox->cufftWrapperDouble(d_dHdwmi, d_dHdwmi, -1);
    mybox->writeTComplexGridData("Fwmi.dat", d_dHdwmi);
    die("done69420!");    


    // Update the fields
    if ( updateScheme == "EM" ) {
        d_fts_updateEM<<<mybox->M_Grid, mybox->M_Block>>>(_d_wpl, _d_dHdwpl, deltPlus, mybox->M);
        d_fts_updateEM<<<mybox->M_Grid, mybox->M_Block>>>(_d_wmi, _d_dHdwmi, deltMinus, mybox->M);
    }

    else if ( updateScheme == "1S" ) {
        // Put the force, potentials into k-space
        mybox->cufftWrapperDouble(d_dHdwpl, d_dHdwpl, 1);
        mybox->cufftWrapperDouble(d_dHdwmi, d_dHdwmi, 1);
        mybox->cufftWrapperDouble(d_wpl, d_wpl, 1);
        mybox->cufftWrapperDouble(d_wmi, d_wmi, 1);



        // Pointers to linear coefficients
        cuDoubleComplex* _d_Akpl = (cuDoubleComplex*)thrust::raw_pointer_cast(d_Akpl.data());
        cuDoubleComplex* _d_Akmi = (cuDoubleComplex*)thrust::raw_pointer_cast(d_Akmi.data());
        
        // Call update scheme
        d_fts_update1S<<<mybox->M_Grid, mybox->M_Block>>>(_d_wpl, _d_dHdwpl, _d_Akpl, deltPlus,  mybox->M);
        d_fts_update1S<<<mybox->M_Grid, mybox->M_Block>>>(_d_wmi, _d_dHdwmi, _d_Akmi, deltMinus, mybox->M);

        // Bring potentials back to r-space
        mybox->cufftWrapperDouble(d_wpl, d_wpl, -1);
        mybox->cufftWrapperDouble(d_wmi, d_wmi, -1);

    }

}

std::complex<double> PotentialFlory::calcHamiltonian() {
    
    thrust::device_vector<thrust::complex<double>> dtmp(mybox->M);

    // dtmp(r) = wpl(r)^2
    thrust::transform(d_wpl.begin(), d_wpl.end(), d_wpl.begin(), dtmp.begin(), 
        thrust::multiplies<thrust::complex<double>>());

    thrust::complex<double> integral = thrust::reduce(dtmp.begin(), dtmp.end()) * mybox->gvol;

    Hterm = integral * mybox->C / chiN ;


    // dtmp(r) = wmi(r)^2
    thrust::transform(d_wmi.begin(), d_wmi.end(), d_wmi.begin(), dtmp.begin(), 
        thrust::multiplies<thrust::complex<double>>());

    integral = thrust::reduce(dtmp.begin(), dtmp.end()) * mybox->gvol;

    Hterm += integral * mybox->C / chiN;

    return Hterm;
}


// This routine is currently written to deal with dHdw in real space
// the I*rho0 term should change if it changed to k-space updating
__global__ void d_makeFloryForce(
    cuDoubleComplex* dHdwpl,              // Field holding dHdwpl
    cuDoubleComplex* dHdwmi,              // Field holding dHdwmi
    const cuDoubleComplex* wpl,           // current d_wpl
    const cuDoubleComplex* wmi,           // current d_wmi
    const cuDoubleComplex* rhoI,        // density of species I
    const cuDoubleComplex* rhoJ,        // density of species J
    const double C,                     // Polymer concentration, based on Nr
    const double chiN,                  // chi * N
    const double Nr,                    // Reference chain length
    const int M                         // number of grid points
    ) {

    const int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= M)
        return;

    // dHdwpl = 2 C wpl / chiN + I * ( rhoI + rhoJ ) / Nr ;
    dHdwpl[ind].x = 2.0 * C * wpl[ind].x / chiN - (rhoI[ind].y + rhoJ[ind].y) / Nr;
    dHdwpl[ind].y = 2.0 * C * wpl[ind].y / chiN + (rhoI[ind].x + rhoJ[ind].x) / Nr;

    // dHdwmi = 2 C wmi / chiN + (rhoJ - rhoI) / Nr;
    // dHdwmi[ind].x = 2.0 * C * wmi[ind].x / chiN + (rhoJ[ind].x - rhoI[ind].x) / Nr;
    // dHdwmi[ind].y = 2.0 * C * wmi[ind].y / chiN + (rhoJ[ind].y - rhoI[ind].y) / Nr;
    dHdwmi[ind].x = (rhoJ[ind].x - rhoI[ind].x) / Nr;
    dHdwmi[ind].y = (rhoJ[ind].y - rhoI[ind].y) / Nr;
}


void PotentialFlory::writeFields(int potInd) {
    char nm[30];
    sprintf(nm, "wpl_Flory%d.dat", potInd);

    // Transfer field to the host;
    wpl = d_wpl;
    mybox->writeTComplexGridData(nm, wpl);


    sprintf(nm, "wmi_Flory%d.dat", potInd);

    // Transfer field to the host;
    wmi = d_wmi;
    mybox->writeTComplexGridData(nm, wmi);
}

void PotentialFlory::initLinearCoeffs() {
    thrust::fill(d_Akpl.begin(), d_Akpl.end(), 2*mybox->C/chiN);
    d_Akmi = d_Akpl;
}

PotentialFlory::~PotentialFlory() {}