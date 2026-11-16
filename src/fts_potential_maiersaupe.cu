// Copyright (c) 2026 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "include_libs.h"
#include "fts_potential_maiersaupe.h"
#include <thrust/inner_product.h>
#include "FTS_Box.h"
#include "fts_species.h"

void die(const char*);

// Input: potential maiersaupe [species] [muN] [delt] (optional arguments)
// Optional arguments:
//   updateScheme EM | EMPC
//   initialize [component] ...   component is xx, xy (2D) or xx, yy, xy, xz, yz (3D),
//                                followed by any of the standard field 
//                                initialization options. The remaining diagonal
//                                component follows from M being traceless.
//                                May be given once per component, but the 
//                                biased-phase options consume the rest of the 
//                                line and so must come last.
PotentialMaierSaupe::PotentialMaierSaupe(std::istringstream& iss, FTS_Box* p_box) : FTS_Potential(iss, p_box) {
    
    // Set stringstream to be ready to read the species
    iss.seekg(0);
    std::string s1;
    iss >> s1;
    iss >> s1;

    potentialStyle = "MaierSaupe";

    const int M = mybox->M;
    const int dim = mybox->returnDimension();
    if ( dim != 2 && dim != 3 ) {
        die("Maier-Saupe FTS potential only set up for 2D and 3D");
    }
    Ntens = dim*(dim+1)/2 - 1;

    // Names of the stored independent components
    if ( dim == 2 ) compNames = {"xx", "xy"};
    else            compNames = {"xx", "yy", "xy", "xz", "yz"};

    iss >> s1;
    actsOn.push_back(s1);

    intSpecies = -1;
    for ( int i=0 ; i<mybox->Species.size() ; i++ ) {
        if ( actsOn[0] == mybox->Species[i].fts_species ) { intSpecies = i; }
    }
    if ( intSpecies < 0 ) {
        std::string LW = actsOn[0] + " species not found for potential MaierSaupe";
        die(LW.c_str());
    }
    if ( mybox->Species[intSpecies].tensor_field != 1 ) {
        std::string LW = "species " + actsOn[0] + " must be declared with the 'tensor' option to use potential MaierSaupe";
        die(LW.c_str());
    }

    iss >> muN;
    iss >> delt;
    std::cout << "Maier-Saupe potential acting on species " << actsOn[0] 
        << " with muN = " << muN << std::endl;

    d_M.resize(Ntens*M, 0.0);
    d_dHdM.resize(Ntens*M, 0.0);

    // One extra slice of noise holds the last diagonal component
    // before the trace is removed
    if ( mybox->ftsStyle == "cl" ) {
        d_noise.resize((Ntens+1)*M, 0.0);
    }
    
    updateScheme = "EM";

    // Host copy used for initialization
    thrust::host_vector<thrust::complex<double>> hM(Ntens*M, 0.0);
    thrust::host_vector<thrust::complex<double>> htmp(M);

    while ( iss >> s1 ) {
        if ( s1 == "initialize" ) {
            iss >> s1;
            int comp = -1;
            for ( int nt=0 ; nt<Ntens ; nt++ ) {
                if ( s1 == compNames[nt] ) comp = nt;
            }
            if ( comp < 0 ) { 
                die("Maier-Saupe initialize expects a tensor component (xx, xy in 2D; xx, yy, xy, xz, yz in 3D) before the initialization style"); 
            }

            thrust::fill(htmp.begin(), htmp.end(), 0.0);
            initializeField(iss, htmp);
            thrust::copy(htmp.begin(), htmp.end(), hM.begin() + comp*M);
        }

        else if ( s1 == "updateScheme" ) {
            iss >> updateScheme;
        }

        else {
            std::string LW = s1 + " is not a valid option for potential MaierSaupe";
            die(LW.c_str());
        }
    }// optional arguments

    if ( updateScheme != "EM" && updateScheme != "EMPC" ) {
        die("Maier-Saupe potential only supports the EM and EMPC update schemes");
    }

    // Allocate storage for predicted forces, fields if doing predictor-corrector scheme
    if ( updateScheme == "EMPC" ) {
        d_dHdMo.resize(Ntens*M, 0.0);
        d_Mo.resize(Ntens*M, 0.0);

        // ensure PC flag set to TRUE.
        mybox->PCflag = true;
    }

    // transfer to device
    d_M = hM;
    
}// PotentialMaierSaupe constructor


// dH/dM = C M / muN + S / Nr for every independent component
void PotentialMaierSaupe::makeForce() {
    const int n = Ntens * mybox->M;
    const int grid = (int)ceil( double(n) / mybox->M_Block );

    cuDoubleComplex* _d_dHdM = (cuDoubleComplex*)thrust::raw_pointer_cast(d_dHdM.data());
    cuDoubleComplex* _d_M = (cuDoubleComplex*)thrust::raw_pointer_cast(d_M.data());
    cuDoubleComplex* _d_S = (cuDoubleComplex*)thrust::raw_pointer_cast(mybox->Species[intSpecies].d_S.data());

    d_makeMaierSaupeForce<<<grid, mybox->M_Block>>>(_d_dHdM, _d_M, _d_S, mybox->C,
        muN, mybox->Nr, n);
}


// Updates potential fields using the chosen scheme
// If two-part predictor/corrector scheme is to be used, then 
// this step is the predictor step
void PotentialMaierSaupe::updateFields() {

    const int M = mybox->M;
    const int n = Ntens * M;
    const int grid = (int)ceil( double(n) / mybox->M_Block );

    bool doCL = false;
    if ( mybox->ftsStyle == "cl" ) doCL = true;

    cuDoubleComplex* _d_dHdM = (cuDoubleComplex*)thrust::raw_pointer_cast(d_dHdM.data());
    cuDoubleComplex* _d_M = (cuDoubleComplex*)thrust::raw_pointer_cast(d_M.data());
    cuDoubleComplex* _d_noise = (cuDoubleComplex*)thrust::raw_pointer_cast(d_noise.data());

    // Generate real, symmetric, traceless tensor noise if doing CL. 
    // The update below moves each stored component with the force taken with
    // respect to the full tensor, dM_ab = -delt * dH/dM_ab + noise_ab. The
    // matching noise is a Gaussian symmetric tensor with its trace removed: 
    // independent diagonal draws of variance 2*delt/gvol minus their mean, and 
    // off-diagonal draws of variance delt/gvol. The RNG states are per grid 
    // point, so the components are drawn one at a time.
    if ( doCL ) {
        const int dim = mybox->returnDimension();
        const int nDiag = dim - 1;                  // stored diagonal components
        const double diagMag = sqrt(2.0 * delt / mybox->gvol );
        const double offMag  = sqrt(delt / mybox->gvol );

        for ( int nt=0 ; nt<Ntens ; nt++ ) {
            d_makeDoubleNoise<<<mybox->M_Grid, mybox->M_Block>>>(_d_noise + nt*M, 
                mybox->d_states, ( nt < nDiag ? diagMag : offMag ), M);
        }

        // Last diagonal component, stored in the extra slice
        d_makeDoubleNoise<<<mybox->M_Grid, mybox->M_Block>>>(_d_noise + Ntens*M, 
            mybox->d_states, diagMag, M);

        d_removeNoiseTrace<<<mybox->M_Grid, mybox->M_Block>>>(_d_noise, nDiag, Ntens, M);
    }

    makeForce();

    if ( updateScheme == "EMPC" ) {
        storePredictorData();
    }

    d_fts_updateEM<<<grid, mybox->M_Block>>>(_d_M, _d_dHdM, _d_noise, doCL, delt, n);

}// updateFields


void PotentialMaierSaupe::correctFields() {
    if ( updateScheme != "EMPC" ) {
        return;
    }

    const int n = Ntens * mybox->M;
    const int grid = (int)ceil( double(n) / mybox->M_Block );

    bool doCL = false;
    if ( mybox->ftsStyle == "cl" ) doCL = true;

    cuDoubleComplex* _d_dHdM = (cuDoubleComplex*)thrust::raw_pointer_cast(d_dHdM.data());
    cuDoubleComplex* _d_M = (cuDoubleComplex*)thrust::raw_pointer_cast(d_M.data());
    cuDoubleComplex* _d_dHdMo = (cuDoubleComplex*)thrust::raw_pointer_cast(d_dHdMo.data());
    cuDoubleComplex* _d_Mo = (cuDoubleComplex*)thrust::raw_pointer_cast(d_Mo.data());
    cuDoubleComplex* _d_noise = (cuDoubleComplex*)thrust::raw_pointer_cast(d_noise.data());

    // Force from the predicted fields
    makeForce();

    d_fts_updateEMPC<<<grid, mybox->M_Block>>>(_d_M, _d_Mo, _d_dHdM, _d_dHdMo, _d_noise, doCL, delt, n);
}


// Stores initial potential and force forms for predictor/corrector updates
void PotentialMaierSaupe::storePredictorData() {
    d_Mo = d_M;
    d_dHdMo = d_dHdM;
}


// Computes this potential's contribution to the effective Hamiltonian,
// C/(2 muN) \int dr M:M
std::complex<double> PotentialMaierSaupe::calcHamiltonian() {
    const int M = mybox->M;
    const int dim = mybox->returnDimension();
    const int nDiag = dim - 1;
    const thrust::complex<double> zero(0.0);

    // sum_r M_a(r) * M_b(r) for stored components a, b
    auto dot = [&](int a, int b) {
        return thrust::inner_product(d_M.begin() + a*M, d_M.begin() + (a+1)*M, 
            d_M.begin() + b*M, zero);
    };

    // Diagonal part, including the component removed by the traceless condition:
    // 2D: Mxx^2 + Myy^2 = 2 Mxx^2
    // 3D: Mxx^2 + Myy^2 + (Mxx + Myy)^2 = 2 ( Mxx^2 + Myy^2 + Mxx Myy )
    thrust::complex<double> sum = zero;
    for ( int a=0 ; a<nDiag ; a++ ) {
        sum += 2.0 * dot(a, a);
        for ( int b=a+1 ; b<nDiag ; b++ ) {
            sum += 2.0 * dot(a, b);
        }
    }

    // Off-diagonal components each appear twice in M:M
    for ( int a=nDiag ; a<Ntens ; a++ ) {
        sum += 2.0 * dot(a, a);
    }

    Hterm = sum * mybox->gvol * mybox->C / 2.0 / muN;

    return Hterm;
}


// Converts independent Gaussian draws for the diagonal of a symmetric tensor
// into a traceless diagonal by subtracting their mean. On input the stored 
// diagonal components are slices [0, nDiag) and the final diagonal component 
// is in slice Ntens.
__global__ void d_removeNoiseTrace(
    cuDoubleComplex* noise,         // [(Ntens+1)*M] noise field
    const int nDiag,                // stored diagonal components, dim-1
    const int Ntens,                // independent tensor components
    const int M                     // grid points
    ) {

    const int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= M)
        return;

    double mean = noise[Ntens*M + ind].x;
    for ( int a=0 ; a<nDiag ; a++ ) {
        mean += noise[a*M + ind].x;
    }
    mean /= double(nDiag + 1);

    for ( int a=0 ; a<nDiag ; a++ ) {
        noise[a*M + ind].x -= mean;
    }
}


// This routine deals with the force in real space
__global__ void d_makeMaierSaupeForce(
    cuDoubleComplex* dHdM,          // [n] force on each tensor component
    const cuDoubleComplex* Mf,      // [n] current tensor field components
    const cuDoubleComplex* S,       // [n] orientation tensor density components
    const double C,                 // Polymer concentration, based on Nr
    const double muN,               // Maier-Saupe strength
    const double Nr,                // Reference chain length
    const int n                     // tensor components * grid points
    ) {

    const int ind = blockIdx.x * blockDim.x + threadIdx.x;
    if (ind >= n)
        return;

    dHdM[ind].x = C * Mf[ind].x / muN + S[ind].x / Nr;
    dHdM[ind].y = C * Mf[ind].y / muN + S[ind].y / Nr;
}


void PotentialMaierSaupe::writeFields(int potInd) {
    const int M = mybox->M;
    char nm[40];

    thrust::host_vector<thrust::complex<double>> hM = d_M;
    thrust::host_vector<thrust::complex<double>> htmp(M);

    for ( int nt=0 ; nt<Ntens ; nt++ ) {
        sprintf(nm, "M%s_MaierSaupe%d.dat", compNames[nt].c_str(), potInd);
        thrust::copy(hM.begin() + nt*M, hM.begin() + (nt+1)*M, htmp.begin());
        mybox->writeTComplexGridData(nm, htmp);
    }
}

// No linear coefficients; semi-implicit updates are not supported
void PotentialMaierSaupe::initLinearCoeffs() {}

PotentialMaierSaupe::~PotentialMaierSaupe() {}
