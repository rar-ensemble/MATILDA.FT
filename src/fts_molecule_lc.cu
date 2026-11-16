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

    if ( Ntot % 2 == 0 ) {
        die("Phase shifts for even numbers of LC sites not set up");
    }

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
        if ( j > 0 && text_species[j] != text_species[j-1] ) {
            die("LC densities not yet set up to handle multiple species");
        }

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

    thrust::device_vector<thrust::complex<double>> t_tmp(M);
    
    cuDoubleComplex* _d_expMddS = (cuDoubleComplex*)thrust::raw_pointer_cast(d_expMddS.data());
    cuDoubleComplex* _d_Mr = (cuDoubleComplex*)thrust::raw_pointer_cast(d_Mr.data());
    double* _d_Stens = (double*)thrust::raw_pointer_cast(mybox->d_Stens.data());
    cuDoubleComplex* _t_tmp = (cuDoubleComplex*)thrust::raw_pointer_cast(t_tmp.data());
    cuDoubleComplex* _d_psi_x = (cuDoubleComplex*)thrust::raw_pointer_cast(d_psi_x.data());
    cuDoubleComplex* _d_psi_y = (cuDoubleComplex*)thrust::raw_pointer_cast(d_psi_y.data());
    cuDoubleComplex* _d_psi_z = (cuDoubleComplex*)thrust::raw_pointer_cast(d_psi_z.data());
    cuDoubleComplex* _d_Nru = (cuDoubleComplex*)thrust::raw_pointer_cast(d_Nru.data());
    cuDoubleComplex* _d_Nr = (cuDoubleComplex*)thrust::raw_pointer_cast(d_Nr.data());

    thrust::fill(d_cStilde.begin(),  d_cStilde.end(),  0.0);
    thrust::fill(d_cDensity.begin(), d_cDensity.end(), 0.0);
    thrust::fill(d_rhou.begin(), d_rhou.end(), 0.0);

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
    d_exp_M_doubledot_S<<<grid, block>>>(_d_expMddS, _d_Mr, _d_Stens, Nu_tot, dim, M);


    // Initialize Nru, Nr 
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

            // phase shift exponent: Ntot/2 - s
            int phase_power = int(Ntot/2) - s;
            d_multiply_phase_shift<<<grid, block>>>(_t_tmp, _t_tmp, _d_psi_x, _d_psi_y, _d_psi_z,
                m, Nu_tot, phase_power, dim, mybox->_d_Nx, M);

            // t_tmp back to real space
            mybox->cufftWrapperDouble(t_tmp, t_tmp, -1);

            // N(r,u) *= E_s(r-b_su, u)
            thrust::transform(t_tmp.begin(), t_tmp.end(), d_Nru.begin() + m*M,
                                d_Nru.begin() + m*M, 
                                thrust::multiplies<thrust::complex<double>>());
        }// s=0:Ntot


        // t_tmp(r) = d_N(r,u_m)
        thrust::copy(d_Nru.begin()+m*M, d_Nru.begin()+(m+1)*M, t_tmp.begin());

        if ( lc_style == "middle" ) {
            // accumulate S_m * N(r,u)
        }

        // t_tmp = N(k,u)
        mybox->cufftWrapperDouble(t_tmp, t_tmp, 1);


        // Sum density contributions for all monomers
        // Gabe = \sum_s \psi_s N(k,u) = R(k,u)
        d_Ntot_phase_shifts<<<grid, block>>>(mybox->d_cpxGabe, _t_tmp, 
            _d_psi_x, _d_psi_y, _d_psi_z,
            m, Nu_tot, Ntot, dim, mybox->_d_Nx, M);

        // Back to real space: t_tmp = R(r,u) = F^(-1)[\sum_s d_Nru]
        mybox->cufftWrapperDouble(mybox->d_cpxGabe, _t_tmp, -1);

        // rho(r,u) = R(r,u)
        thrust::copy(t_tmp.begin(), t_tmp.end(), d_rhou.begin() + m*M);

        if ( lc_style == "all" ) {
            // accumulate S_m * R(r,u)
        }

    }// m=0:Nu_tot
    
    mybox->integrate_u_fixed_r(_d_Nr, _d_Nru);

    thrust::complex<double> prefactor = nmolecs * mybox->gvol / mybox->V / PI2;
    
    // Angle normalization should be 4PI in 3D
    if ( dim == 3 ) prefactor *= 0.5;

    Q = thrust::reduce(d_Nr.begin(), d_Nr.end(), thrust::complex<double>(0.0), 
                        thrust::plus<thrust::complex<double>>()) * prefactor;

    // Multiply rho(r,u) by n/Omega/V/Q
    prefactor *= 1.0 / Q;
    thrust::transform( d_rhou.begin(), d_rhou.end(), d_rhou.begin(),
        [=] __device__ (thrust::complex<double> x) {
            return x * prefactor;
        }
    );
        
    // accumulate rho(r) = \int du rho(r,u)
    cuDoubleComplex *_d_cDensity = 
            (cuDoubleComplex*) thrust::raw_pointer_cast(d_cDensity.data());
    cuDoubleComplex *_d_rhou = 
            (cuDoubleComplex*) thrust::raw_pointer_cast(d_rhou.data());
    mybox->integrate_u_fixed_r(_d_cDensity, _d_rhou);


    // Finally, smear the relevant fields if needed
    if ( doSmear ) {
        mybox->convolveTComplexDouble(d_cDensity, t_tmp, d_smearFunc);
        // Make this convolve S field
        // mybox->convolveTComplexDouble(d_cDensity, t_tmp, d_smearFunc);
    }

    // Not smearing, so just equate the fields
    else {
        d_density = d_cDensity;
        d_Stilde = d_cStilde;
    }


}
// gabe here :3

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
        int nn[3];
        mybox->unstack2(i, nn);

        for ( int j=0 ; j<Nu ; j++ ) {
            if ( Dim == 2 ) {
                double theta = dth * i;
                double ux = mybox->uvecs[2*j + 0];
                double uy = mybox->uvecs[2*j + 1];

                psi_x[nn[0]*Nu + j] = exp(-I * bond_length * kv[0]*ux);
                psi_y[nn[1]*Nu + j] = exp(-I * bond_length * kv[1]*uy);
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
