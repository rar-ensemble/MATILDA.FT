// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "ps_potential.h"
#include "ps_potentialGaussian.h"
#include "ps_potentialErf.h"
#include "PS_Box.h"


// Allocates memory for:
// potential, forces, virial contribution
// in both r- and k-space
void PS_Potential::initializePotential() {
    
    Iind = mybox->findGroupInteger(grpI);
    Jind = mybox->findGroupInteger(grpJ);

    // Allocate grid force memory for groups I, J if needed
    if ( mybox->psGroup[Iind].hasForce() == 0 ) { mybox->psGroup[Iind].enableForce(); }
    if ( mybox->psGroup[Jind].hasForce() == 0 ) { mybox->psGroup[Jind].enableForce(); }

}



// Calculates forces on rho1, rho2 for this pairstyle
// This virtual function should be overridden by classes that
// either don't use a scalar potential (e.g., Maier-Saupe) or
// use a higher-than-two-body potentials
void PS_Potential::CalcForces() {

    float *d_rhoI, *d_rhoJ;

    // Pointers to reused box vars
    cuComplex *d_cpxAlex, *d_cpxGabe;
    float *d_Gabe;
    d_cpxAlex = mybox->d_cpxAlex;
    d_cpxGabe = mybox->d_cpxGabe;
    d_Gabe = mybox->d_Gabe;
    // float *d_Alex;
    // d_Alex = mybox->d_Alex;

    int M = mybox->M;
    int Dim = mybox->returnDimension();
    int Grid = mybox->M_Grid;
    int Block = mybox->M_Block;


    ///////////////////////////////////////////////
    // Forces acting on type I arise from type J //
    ///////////////////////////////////////////////

    // Pointer to density field for J
    d_rhoJ = mybox->psGroup[Jind].d_rho;

    // real(Alex) = rhoJ, imag(Alex) = 0.0
    d_floatToCpx<<<Grid, Block>>>(d_cpxAlex, d_rhoJ, M);

    // Gabe = FT(Alex=rhoJ); Alex now available
    mybox->cufftWrapperSingle(d_cpxAlex, d_cpxGabe, 1);

    check_cudaError("Potential first fft");
    

    for ( int j=0 ; j<Dim ; j++ ) {
        // Alex = fk[j] * FT(rhoJ), j \in [x,y,z]
        d_multiplyCpxDirByCpx<<<Grid, Block>>>(d_cpxAlex, d_fk, d_cpxGabe, j, Dim, M);


        // Gabe = IFT(Alex)
        mybox->cufftWrapperSingle(d_cpxAlex, d_cpxAlex, -1);
        d_cpxToFloat<<<Grid, Block>>>(d_Gabe, d_cpxAlex, M);


        // Gabe now contains the forces that act on particles I
        mybox->psGroup[Iind].accumulateGridForceComp(d_Gabe, j);

        // If the group acts on itself, simply accumulate the grid force twice
        if ( Iind == Jind ) {
            // Gabe now contains the forces that act on particles I
            mybox->psGroup[Iind].accumulateGridForceComp(d_Gabe, j);
        }    
    }
    check_cudaError("Potential first force accumulation");

    // Group does not act on itself // 
    if ( Iind != Jind ) {
        
        ///////////////////////////////////////////////
        // Forces acting on type J arise from type I //
        ///////////////////////////////////////////////
        // Pointer to density field for J
        d_rhoI = mybox->psGroup[Iind].d_rho;

        // real(Alex) = rhoI, imag(Alex) = 0.0
        d_floatToCpx<<<Grid, Block>>>(d_cpxAlex, d_rhoI, M);

        // Gabe = FT(Alex); Alex now available
        mybox->cufftWrapperSingle(d_cpxAlex, d_cpxGabe, 1);

        for ( int j=0 ; j<Dim ; j++ ) {
            // Alex = fk[j] * FT(rhoJ), j \in [x,y,z]
            d_multiplyCpxDirByCpx<<<Grid, Block>>>(d_cpxAlex, d_fk, d_cpxGabe, j, Dim, M);

            // Gabe = IFT(Alex)
            mybox->cufftWrapperSingle(d_cpxAlex, d_cpxAlex, -1);
            d_cpxToFloat<<<Grid, Block>>>(d_Gabe, d_cpxAlex, M);

            // Gabe now contains the forces that act on particles J
            mybox->psGroup[Jind].accumulateGridForceComp(d_Gabe, j);
        }
    }

}



// Calculates the energy involved in this potential as
// energy = \int dr rho1(r) \int dr' u(r-r') rho2(r')
// The convolution theorem is used to efficiently evaluate
// the integral over r'
float PS_Potential::CalcEnergy() {
    energy = 0.0f;

    float *d_rhoI, *d_rhoJ;

    // Pointers to reused box vars
    cuComplex *d_cpxAlex, *d_cpxGabe;
    float *d_Gabe, *d_Alex;
    d_cpxAlex = mybox->d_cpxAlex;
    d_cpxGabe = mybox->d_cpxGabe;
    d_Gabe = mybox->d_Gabe;
    d_Alex = mybox->d_Alex;

    int M = mybox->M;
    int Dim = mybox->returnDimension();
    int Grid = mybox->M_Grid;
    int Block = mybox->M_Block;


    // Pointer to density field for J
    d_rhoJ = mybox->psGroup[Jind].d_rho;

    // real(Alex) = rhoJ, imag(Alex) = 0.0
    d_floatToCpx<<<Grid, Block>>>(d_cpxAlex, d_rhoJ, M);

    // Gabe = FT(Alex=rhoJ); Alex now available
    mybox->cufftWrapperSingle(d_cpxAlex, d_cpxGabe, 1);

    // Alex = uk[j] * FT(rhoJ), j \in [x,y,z]
    d_multiplyCpxByCpx<<<Grid, Block>>>(d_cpxAlex, d_uk, d_cpxGabe, M);


    // Gabe = IFT(Alex) = [u \ast \rho_J](r)
    mybox->cufftWrapperSingle(d_cpxAlex, d_cpxAlex, -1);
    d_cpxToFloat<<<Grid, Block>>>(d_Gabe, d_cpxAlex, M);

    // Alex = rhoI * Gabe
    d_rhoI = mybox->psGroup[Iind].d_rho;
    d_multiplyFloatByFloat<<<Grid, Block>>>(d_Alex, d_rhoI, d_Gabe, M);

    energy = mybox->gvol * mybox->sumDeviceArray(d_Alex, 
                                mybox->M_Block, mybox->M);

    return this->energy;
    
}






PS_Potential::~PS_Potential() {


}

PS_Potential::PS_Potential() {
    return;
}

PS_Potential::PS_Potential(std::istringstream &iss, PS_Box* box) : mybox(box) {
    input_command = iss.str();

    int M = mybox->M;
    int Dim = mybox->returnDimension();
    int nPC = mybox->n_P_comps;

    // Host real-space variables
    ur = (float*) malloc( M * sizeof(float));
    fI = (float*) malloc( M*Dim * sizeof(float));
    fJ = (float*) malloc( M*Dim * sizeof(float));

    // Host k-space variables
    uk = (std::complex<float>*) malloc(M * sizeof(std::complex<float>));
    fk = (std::complex<float>*) malloc( M*Dim * sizeof(std::complex<float>));
    virk = (std::complex<float>*) malloc( M*nPC * sizeof(std::complex<float>));

    // Device real-space variables
    cudaMalloc(&d_ur, M * sizeof(float));
    cudaMalloc(&d_fI, M*Dim * sizeof(float));
    cudaMalloc(&d_fJ, M*Dim * sizeof(float));

    // Device k-space variables
    cudaMalloc(&d_uk,   M * sizeof(cuComplex));
    cudaMalloc(&d_fk,   M*Dim * sizeof(cuComplex));
    cudaMalloc(&d_virk, M*nPC * sizeof(cuComplex));
    cudaMalloc(&d_fI,   M*Dim * sizeof(cuComplex));
    cudaMalloc(&d_fJ,   M*Dim * sizeof(cuComplex));
    
    return;
}


void PS_Potential::ramp_check_input(std::istringstream& iss){

    // if (iss.fail()){
    //     die("Error during input script; failed to properly read:\n" + iss.str());
    // }

    // string convert;
    // iss >> convert;

    // if (!iss.fail()){
    //     if (convert == "ramp") {
    //         ramp = true;
    //         iss >> final_prefactor;
    //         if(iss.fail()) die("no final prefactor specified");
    //         cout << "Ramping prefactor of " <<potential_type<< " style, between types " << type1+1 << " and " \
    //             << type2 + 1 << " from " << initial_prefactor \
    //             << " to " << final_prefactor << endl;

    //         cout << "Estimated per time step change: " << \
    //             (final_prefactor - initial_prefactor) / (prod_steps)
    //             << endl;

    //     }
    //     else 
    //         die("Invalid keyword: " + convert);
    // }


}


PS_Potential* PSPotentialFactory(std::istringstream &iss, PS_Box* box){
 	std::string s1;
 	iss >> s1;
    
    // std::transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
// 	if (s1 == "erf"){
// 		return new Erf(iss);
// 	}
 	if (s1 == "gaussian"){
 		return new NBGauss(iss, box);
 	}
    else if ( s1 == "erf" ) {
        return new NBErf(iss,box);
    }
// 	if (s1 == "gaussian_erf"){
// 		return new GaussianErf(iss);
// 	}
// 	if (s1 == "fieldphase" || s1 == "biasfield"){
// 		return new BiasField(iss);
// 	}
// 	if (s1 == "maiersaupe"){
// 		return new MaierSaupe(iss);
// 	}
// 	if (s1 == "charges"){
// 		return new Charges(iss);
// 	}
	
 	die("Unsupported potential");
 	return 0;
 }
