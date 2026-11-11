// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "fts_molecule_lc.h"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>
#include <thrust/copy.h>
#include "FTS_Box.h"
#include "fts_species.h"
void die(const char*);

LCMolec::~LCMolec(){}

LCMolec::LCMolec(std::istringstream& iss, FTS_Box* p_box) : FTS_Molec(iss, p_box) {
    if ( mybox->Nu_tot == 0 ) {
        die("must define u_grid in input file before initializing liquid crystal molecule");
    }

    // Number of sites on the LC
    iss >> Ntot ;

    bond_length = 1.0;
    lc_style = "middle";

    // Allocate the site-specific vectors
    text_species.resize(Ntot);
    int_species.resize(Ntot);
    tensor_flag.resize(Ntot, 0);
    d_int_species.resize(Ntot);

    for ( int i=0 ; i<Ntot ; i++ ) {
        iss >> text_species[i] ;
    }

    // Optional arguments
    while ( iss.tellg() != -1 ) {
        std::string s1;
        iss >> s1;
        if ( s1 == "smear" || s1 == "Smear" ) {
            iss >> s1 ; 
            doSmear = 1;
            if ( s1 == "Gaussian" || s1 == "gaussian" ) {
                iss >> smearLength;
                std::cout << "Smearing with unit Gaussian with width " << smearLength << std::endl;
            }
            else {
                die("invalid smearing function passed to liquid-crystal molecule");
            }
        } // s1 == "smear"

        else if ( s1 == "lc_style" || s1 == "lcStyle" ) {
            iss >> lc_style;
            if ( lc_style != "all" && lc_style != "middle" ) {
                die("Invalid lc_style given for liquid-crystal molecule");
            }
            if ( lc_style == "middle" && Ntot%2 == 0 ) {
                die("LCMolec can't handle even-numbers of sites with lc_style middle");
            }
        }

        else if ( s1 == "bond_length" || s1 == "bondLength" ) {
            iss >> bond_length;
        }
    } // end of optional arguments

    // Set flags for which particles carry orientation interactions
    if ( lc_style == "middle" ) {
        tensor_flag[ int(Ntot/2) ] = 1;
    }
    else if (lc_style == "all" ) {
        for ( int s=0 ; s<Ntot ; s++ ) tensor_flag[s] = 1;
    }


    // Determine integer block Species
    for ( int j=0 ; j<Ntot; j++ ) {
        int_species[j] = -55;
        for ( int i=0 ; i<mybox->Species.size(); i++ ) {
            if ( text_species[j] == mybox->Species[i].fts_species ) {
                int_species[j] = i;
            }
        }
        if ( int_species[j] < 0 ) {
            std::string lastWords = text_species[j] + " species not found!";
            die(lastWords.c_str());
        }
    }
    // Send integers to device
    d_int_species = int_species;



    // Allocate field variables
    // scalar fields
    int M = mybox->M;
    int Nru = M * mybox->Nu_tot;
    int Ntens;
    if ( mybox->returnDimension() == 2 ) {
        Ntens = 2 * M;
    }
    else if ( mybox->returnDimension() == 3 ) {
        Ntens = 5 * M;
    }

    // Allocate storage for scalar field
    d_W.resize(M);
    d_E.resize(Ntot * M);
    d_expW.resize(M);
    
    // M(r) and Stilde fields
    d_Mr.resize(Ntens);

    Stilde.resize(Ntens);
    d_Stilde.resize(Ntens);
    
    cStilde.resize(Ntens);
    d_cStilde.resize(Ntens);

    // functions of (r, u)
    rhou.resize(Nru);
    d_rhou.resize(Nru);
    d_Nru.resize(Nru);
    d_expMddS.resize(Nru);
    
    // int du N(r,u)
    d_Nr.resize(M);


    // Allocate smearing
    smearFunc.resize(M);
    d_smearFunc.resize(M);
    
    
    // initializes Gaussian smear in k-space
    mybox->initSmearGaussian(smearFunc, 1.0, smearLength);
    
    // Send smear to the device
    d_smearFunc = smearFunc;

}//LCMolec constructor

struct NegExponential {
    __host__ __device__
        thrust::complex<double> operator()(const thrust::complex<double> &z) {
            return exp(-z);
    }
};

void LCMolec::calcDensity() {
    int Nu_tot = mybox->Nu_tot;
    int dim = mybox->returnDimension();
    int M = mybox->M;
    int grid = mybox->M_Grid;
    int block = mybox->M_Block;
    double dth = 2.0 * PI / double(mybox->Nu[0]);
    int grid = mybox->M_Grid;
    int block = mybox->M_Block;


    double Lc = bond_length * (Ntot - 1);

    thrust::fill(d_cStilde.begin(),  d_cStilde.end(),  0.0);
    thrust::fill(d_cDensity.begin(), d_cDensity.end(), 0.0);
    
    thrust::device_vector<thrust::complex<double>> t_tmp(M);

    // Transform the scalar field to generate E(k)
    // Also smear M(r) if needed
    for ( int s=0 ; s<Ntot ; s++ ) {
        
        // Convolve fields with smearing function as needed
        if ( doSmear ) {
            mybox->convolveTComplexDouble(mybox->Species[int_species[s]].d_w, 
                d_W, d_smearFunc);

            // Smear independent components of M tensor
            if ( mybox->Species[int_species[s]].tensor_field == 1 ) {
                int Ntens = mybox->Ntens;
                for ( int nt=0 ; nt<Ntens ; nt++ ) {
                    thrust::copy(mybox->Species[int_species[s]].d_Mw.begin(), 
                        mybox->Species[int_species[s]].d_Mw.end(), 
                        t_tmp.begin());

                    mybox->convolveTComplexDouble(t_tmp, t_tmp, d_smearFunc);

                    thrust::copy(t_tmp.begin(), t_tmp.end(), 
                        d_Mr.begin() + nt*M);
                }
            }
        }

        // If not smeared just copy fields
        else {
            d_W = mybox->Species[int_species[s]].d_w;
            if ( mybox->Species[int_species[s]].tensor_field == 1 ) {
                d_Mr = mybox->Species[int_species[s]].d_Mw;
            }
        }
    
        // expW = exp(-W)
        thrust::transform(d_W.begin(), d_W.end(), d_expW.begin(), NegExponential());

        // d_E(r,s) = expW(r), offset by s*M to store in different regions of array
        thrust::copy(d_expW.begin(), d_expW.end(), d_E.begin()+s*M);
    }// s=0:Ntot
    

    // Computes exp_M_doubledot_S = exp(-M(r):S(u)) for all r, u
    cuDoubleComplex* _d_expMddS = (cuDoubleComplex*)thrust::raw_pointer_cast(d_expMddS.data());
    cuDoubleComplex* _d_Mr = (cuDoubleComplex*)thrust::raw_pointer_cast(d_Mr.data());
    double* _d_Stens = (double*)thrust::raw_pointer_cast(mybox->d_Stens.data());

    d_exp_M_doubledot_S<<<grid, block>>>(_d_expMddS, _d_Mr, _d_Stens, Nu_tot, dim, M);


    // Zero 
    thrust::fill(d_Nru.begin(), d_Nru.end(), 1.0);
    thrust::fill(d_Nr.begin(), d_Nr.end(), 0.0);

    for ( int m=0 ; m<Nu_tot ; m++ ) { 
        for ( int s=0 ; s<Ntot ; s++ ) { 

            if ( tensor_flag[s] == 1 ) {
                // t_tmp(r) = exp(-M(r):S)*exp(-w(r))
                thrust::transform(d_expMddS.begin() + m*M, d_expMddS.begin() + (m+1)*M, 
                    d_E.begin() + s*M, t_tmp.begin(), thrust::multiplies<thrust::complex<double>>());
            }

            else { 
                // t_tmp(r) = exp(-w(r))
                thrust::copy(d_E.begin() + s*M, d_E.begin() + (s+1)*M, t_tmp.begin());
            }

            // Compute F.T. of exp(-w-M:S)
            mybox->cufftWrapperDouble(t_tmp, t_tmp, 1);

            

        }// s=0:Ntot
    }// m=0:Nu_tot
    
}


void LCMolec::finish_initialization() {
    int Nu = mybox->Nu_tot;
    int M = mybox->M;
    int Dim = mybox->returnDimension();
    double dth = 2.0 * PI / double(mybox->Nu[0]);
    std::complex<double> I(0.0, 1.0);

    psi_x.resize(Nu*mybox->Nx[0]);
    psi_y.resize(Nu*mybox->Nx[1]);
    if ( Dim == 3 ) psi_z.resize(Nu*mybox->Nx[2]);

    for ( int i=0 ; i<M ; i++ ) {
        double kv[3];
        (void) mybox->get_kD(i, kv);

        for ( int j=0 ; j<Nu ; j++ ) {
            if ( Dim == 2 ) {
                double theta = dth * i;
                double ux = mybox->uvecs[2*j + 0];
                double uy = mybox->uvecs[2*j + 1];

                psi_x[i*Nu + j] = exp(-I * bond_length * kv[0]*ux);
                psi_y[i*Nu + j] = exp(-I * bond_length * kv[1]*uy);
            }
        }// j=0:Nu
    }

    d_psi_x.resize(Nu*mybox->Nx[0]);
    d_psi_y.resize(Nu*mybox->Nx[1]);
    if ( Dim == 3 ) d_psi_z.resize(Nu*mybox->Nx[2]);

    d_psi_x = psi_x;
    d_psi_y = psi_y;
    if ( Dim == 3 ) d_psi_z = psi_z;

    std::cout << "    psi_\alpha variables initialized for LCMolec" << std::endl;
}


// Recomputes number of molecules for GCE simulations
void LCMolec::recomputeNmolecs() {
    if ( activity > 0.0 ) {
        nmolecs = mybox->C * phi * mybox->Vfree * mybox->Nr / double(Ntot);   
    }
}

void LCMolec::computeLinearTerms() {
}
