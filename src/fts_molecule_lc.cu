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
    doSmear = 0;
    smearLength = 0.0;

    // Allocate the site-specific vectors
    text_species.resize(Ntot);
    int_species.resize(Ntot);
    tensor_flag.resize(Ntot, 0);
    d_int_species.resize(Ntot);

    for ( int i=0 ; i<Ntot ; i++ ) {
        iss >> text_species[i] ;
    }

    // Optional arguments
    std::string s1;
    while ( iss >> s1 ) {
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
        }

        else if ( s1 == "bond_length" || s1 == "bondLength" ) {
            iss >> bond_length;
        }

        else {
            std::string lastWords = s1 + " is not a valid option for a liquid-crystal molecule";
            die(lastWords.c_str());
        }
    } // end of optional arguments

    // Set flags for which particles carry orientation interactions
    if ( lc_style == "middle" ) {
        tensor_flag[ int(Ntot/2) ] = 1;
    }
    else if (lc_style == "all" ) {
        for ( int s=0 ; s<Ntot ; s++ ) tensor_flag[s] = 1;
    }


    // Determine integer species of each site. The sites may be of different
    // species; each unique species gets a "slot" for its fields and operators.
    site_slot.resize(Ntot);
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

        site_slot[j] = -1;
        for ( int k=0 ; k<unique_species.size() ; k++ ) {
            if ( unique_species[k] == int_species[j] ) site_slot[j] = k;
        }
        if ( site_slot[j] < 0 ) {
            unique_species.push_back(int_species[j]);
            site_slot[j] = unique_species.size() - 1;
        }
    }
    nUnique = unique_species.size();

    // Send integers to device
    d_int_species = int_species;

    // Masks picking out the sites of each unique species
    thrust::host_vector<int> site_mask(nUnique*Ntot, 0);
    for ( int j=0 ; j<Ntot ; j++ ) {
        site_mask[ site_slot[j]*Ntot + j ] = 1;
    }
    d_site_mask = site_mask;

    // Which species need exp(-M:S), and which need a stored transform of 
    // exp(-w) (those with a translated site that has no tensor field)
    const int mid = int(Ntot/2);
    tens_slot.assign(nUnique, -1);
    needs_Ek.assign(nUnique, 0);
    nTensSlots = 0;
    for ( int j=0 ; j<Ntot ; j++ ) {
        if ( tensor_flag[j] == 1 ) {
            // The tensor field is only applied if the species carries one
            if ( mybox->Species[int_species[j]].tensor_field != 1 ) {
                std::string lastWords = "species " + text_species[j] 
                    + " must be declared with the 'tensor' option to carry orientation interactions";
                die(lastWords.c_str());
            }
            if ( tens_slot[site_slot[j]] < 0 ) {
                tens_slot[site_slot[j]] = nTensSlots++;
            }
        }
        else if ( j != mid ) {
            needs_Ek[site_slot[j]] = 1;
        }
    }


    // Allocate field variables
    int M = mybox->M;
    int Nru = M * mybox->Nu_tot;
    int dim = mybox->returnDimension();
    int Ntens = (dim*(dim+1)/2 - 1) * M;

    // Density of each unique species (base class allocates only [M])
    density.resize(nUnique*M);
    d_density.resize(nUnique*M);
    d_cDensity.resize(nUnique*M);

    // Allocate storage for scalar field
    d_W.resize(M);
    d_E.resize(nUnique * M);
    d_Ek.resize(nUnique * M);
    d_tmp.resize(M);
    
    // M(r) and Stilde fields
    d_Mr.resize(Ntens);

    Stilde.resize(nUnique*Ntens);
    d_Stilde.resize(nUnique*Ntens);
    
    cStilde.resize(nUnique*Ntens);
    d_cStilde.resize(nUnique*Ntens);

    // functions of (r, u)
    d_rhou.resize(nUnique*Nru);
    d_Nru.resize(Nru);
    d_expMddS.resize(nTensSlots*Nru);
    
    // int du N(r,u)
    d_Nr.resize(M);


    // Allocate smearing
    if ( doSmear ) {
        smearFunc.resize(M);
        d_smearFunc.resize(M);
        
        // initializes Gaussian smear in k-space
        mybox->initSmearGaussian(smearFunc, 1.0, smearLength);
        
        // Send smear to the device
        d_smearFunc = smearFunc;
    }

}//LCMolec constructor

struct NegExponential {
    __host__ __device__
        thrust::complex<double> operator()(const thrust::complex<double> &z) {
            return exp(-z);
    }
};

void LCMolec::calcDensity() {
    const int Nu_tot = mybox->Nu_tot;
    const int dim = mybox->returnDimension();
    const int M = mybox->M;
    const int grid = mybox->M_Grid;
    const int block = mybox->M_Block;
    const int Ntens = dim*(dim+1)/2 - 1;
    const int mid = int(Ntot/2);

    // Orientation quadrature weight and total solid angle
    double du, Omega;
    if ( dim == 2 ) {
        Omega = 2.0 * M_PI;
        du = Omega / double(Nu_tot);
    }
    else {
        die("LCMolec::calcDensity only set up for 2D");
        return;
    }

    cuDoubleComplex* _d_expMddS = (cuDoubleComplex*)thrust::raw_pointer_cast(d_expMddS.data());
    cuDoubleComplex* _d_Mr = (cuDoubleComplex*)thrust::raw_pointer_cast(d_Mr.data());
    double* _d_Stens = (double*)thrust::raw_pointer_cast(mybox->d_Stens.data());
    cuDoubleComplex* _d_tmp = (cuDoubleComplex*)thrust::raw_pointer_cast(d_tmp.data());
    cuDoubleComplex* _d_psi_x = (cuDoubleComplex*)thrust::raw_pointer_cast(d_psi_x.data());
    cuDoubleComplex* _d_psi_y = (cuDoubleComplex*)thrust::raw_pointer_cast(d_psi_y.data());
    cuDoubleComplex* _d_psi_z = (cuDoubleComplex*)thrust::raw_pointer_cast(d_psi_z.data());
    cuDoubleComplex* _d_Nru = (cuDoubleComplex*)thrust::raw_pointer_cast(d_Nru.data());
    cuDoubleComplex* _d_Nr = (cuDoubleComplex*)thrust::raw_pointer_cast(d_Nr.data());
    cuDoubleComplex* _d_Ek = (cuDoubleComplex*)thrust::raw_pointer_cast(d_Ek.data());
    cuDoubleComplex* _d_cStilde = (cuDoubleComplex*)thrust::raw_pointer_cast(d_cStilde.data());
    cuDoubleComplex* _d_cDensity = (cuDoubleComplex*)thrust::raw_pointer_cast(d_cDensity.data());
    cuDoubleComplex* _d_rhou = (cuDoubleComplex*)thrust::raw_pointer_cast(d_rhou.data());
    int* _d_site_mask = (int*)thrust::raw_pointer_cast(d_site_mask.data());

    thrust::fill(d_cStilde.begin(),  d_cStilde.end(),  0.0);
    thrust::fill(d_cDensity.begin(), d_cDensity.end(), 0.0);

    //////////////////////////////////////////////////////////////
    // Fields for each unique species j on the mesogen:         //
    // E_j(r) = exp(-w_j) and, if needed, exp(-M_j(r):S(u))     //
    //////////////////////////////////////////////////////////////
    for ( int j=0 ; j<nUnique ; j++ ) {
        FTS_Species &sp = mybox->Species[unique_species[j]];

        // Convolve field with smearing function as needed
        if ( doSmear ) {
            mybox->convolveTComplexDouble(sp.d_w, d_W, d_smearFunc);
        }
        else {
            d_W = sp.d_w;
        }
    
        // d_E(r,j) = exp(-W), offset by j*M to store in different regions of array
        thrust::transform(d_W.begin(), d_W.end(), d_E.begin() + j*M, NegExponential());

        // Sites with no orientation dependence only need one transform:
        // E_j(k) * psi^(mid-s) is all that changes with u
        if ( needs_Ek[j] ) {
            thrust::copy(d_E.begin() + j*M, d_E.begin() + (j+1)*M, d_tmp.begin());
            mybox->cufftWrapperDouble(d_tmp, d_tmp, 1);
            thrust::copy(d_tmp.begin(), d_tmp.end(), d_Ek.begin() + j*M);
        }

        // exp(-M(r):S(u)) for all r, u, using this species' tensor field
        if ( tens_slot[j] >= 0 ) {

            // Smear each independent component of M
            if ( doSmear ) {
                for ( int nt=0 ; nt<Ntens ; nt++ ) {
                    thrust::copy(sp.d_Mw.begin() + nt*M, sp.d_Mw.begin() + (nt+1)*M, d_tmp.begin());

                    mybox->convolveTComplexDouble(d_tmp, d_tmp, d_smearFunc);

                    thrust::copy(d_tmp.begin(), d_tmp.end(), d_Mr.begin() + nt*M);
                }
            }
            else {
                d_Mr = sp.d_Mw;
            }

            d_exp_M_doubledot_S<<<grid, block>>>(_d_expMddS + tens_slot[j]*Nu_tot*M, 
                _d_Mr, _d_Stens, Nu_tot, dim, M);
        }
    }// j=0:nUnique
    

    /////////////////////////////////////////////////////
    // N(r,u) = prod_s E_s(r + (s-mid) b u, u), R(r,u) //
    /////////////////////////////////////////////////////
    thrust::fill(d_Nru.begin(), d_Nru.end(), 1.0);

    for ( int m=0 ; m<Nu_tot ; m++ ) { 
        for ( int s=0 ; s<Ntot ; s++ ) { 
            const int j = site_slot[s];

            // phase shift exponent: psi^p translates a field by +p*b*u, 
            // f(r) -> f(r - p b u)
            int phase_power = mid - s;

            if ( tensor_flag[s] == 1 ) {
                // d_tmp(r) = exp(-M(r):S)*exp(-w(r))
                const int off = (tens_slot[j]*Nu_tot + m) * M;
                thrust::transform(d_expMddS.begin() + off, d_expMddS.begin() + off + M, 
                    d_E.begin() + j*M, d_tmp.begin(), thrust::multiplies<thrust::complex<double>>());
            }

            // Sites needing a translation
            if ( phase_power != 0 ) {
                const cuDoubleComplex* _src;

                if ( tensor_flag[s] == 1 ) {
                    mybox->cufftWrapperDouble(d_tmp, d_tmp, 1);
                    _src = _d_tmp;
                }
                else {
                    _src = _d_Ek + j*M;
                }

                d_multiply_phase_shift<<<grid, block>>>(_d_tmp, _src, _d_psi_x, _d_psi_y, _d_psi_z,
                    m, Nu_tot, phase_power, dim, mybox->_d_Nx, M);

                // d_tmp back to real space
                mybox->cufftWrapperDouble(d_tmp, d_tmp, -1);
            }

            // Untranslated site with no tensor field
            else if ( tensor_flag[s] == 0 ) {
                thrust::copy(d_E.begin() + j*M, d_E.begin() + (j+1)*M, d_tmp.begin());
            }

            // N(r,u) *= E_s(r + (s-mid) b u, u)
            thrust::transform(d_tmp.begin(), d_tmp.end(), d_Nru.begin() + m*M,
                                d_Nru.begin() + m*M, 
                                thrust::multiplies<thrust::complex<double>>());
        }// s=0:Ntot


        // Only the central site carries orientation: S(r) += du * S(u_m) N(r,u_m),
        // credited to the central site's species
        if ( lc_style == "middle" ) {
            d_accumulate_S<<<grid, block>>>(_d_cStilde + site_slot[mid]*Ntens*M, 
                _d_Nru + m*M, _d_Stens, m, Ntens, du, M);
        }

        // d_tmp = N(k,u)
        thrust::copy(d_Nru.begin()+m*M, d_Nru.begin()+(m+1)*M, d_tmp.begin());
        mybox->cufftWrapperDouble(d_tmp, d_tmp, 1);

        // Density of each unique species: sum the translated center
        // distribution over the sites of that species
        for ( int j=0 ; j<nUnique ; j++ ) {
            cuDoubleComplex* _rhou_jm = _d_rhou + (j*Nu_tot + m)*M;

            // Gabe = \sum_{s in j} \psi^(s-mid) N(k,u) = R_j(k,u)
            d_Ntot_phase_shifts<<<grid, block>>>(mybox->d_cpxGabe, _d_tmp, 
                _d_psi_x, _d_psi_y, _d_psi_z, _d_site_mask + j*Ntot,
                m, Nu_tot, Ntot, dim, mybox->_d_Nx, M);

            // Back to real space: rho_j(r,u) = R_j(r,u)
            mybox->cufftWrapperDouble(mybox->d_cpxGabe, _rhou_jm, -1);

            // All sites carry orientation: S_j(r) += du * S(u_m) R_j(r,u_m)
            if ( lc_style == "all" ) {
                d_accumulate_S<<<grid, block>>>(_d_cStilde + j*Ntens*M, _rhou_jm, 
                    _d_Stens, m, Ntens, du, M);
            }
        }

    }// m=0:Nu_tot
    

    ///////////////////////////////////////
    // Partition function and prefactors //
    ///////////////////////////////////////

    // N(r) = \int du N(r,u)
    mybox->integrate_u_fixed_r(_d_Nr, _d_Nru);

    // Q = 1/(Omega V) \int dr N(r)
    Q = thrust::reduce(d_Nr.begin(), d_Nr.end(), thrust::complex<double>(0.0), 
                        thrust::plus<thrust::complex<double>>()) * mybox->gvol / mybox->V / Omega;

    // Operator prefactor: n/(Omega V Q) in the canonical ensemble, z/Omega if
    // an activity is specified
    thrust::complex<double> prefactor;
    if ( nmolecs > 0.0 ) {
        prefactor = nmolecs / Omega / mybox->V / Q;
    }
    else if ( activity > 0.0 ) {
        prefactor = activity / Omega;
        nSites = activity * mybox->V * Q.real() * Ntot;
    }
    else {
        die("LCMolec needs either nmolecs > 0 or activity > 0");
    }

    auto scale = [=] __device__ (thrust::complex<double> x) -> thrust::complex<double> { return x * prefactor; };
    thrust::transform( d_rhou.begin(), d_rhou.end(), d_rhou.begin(), scale );
    thrust::transform( d_cStilde.begin(), d_cStilde.end(), d_cStilde.begin(), scale );
        
    //////////////////////////////////////////////////////////
    // Integrate, smear, and pass operators to each species //
    //////////////////////////////////////////////////////////
    for ( int j=0 ; j<nUnique ; j++ ) {
        FTS_Species &sp = mybox->Species[unique_species[j]];

        // center density rho_j(r) = \int du rho_j(r,u)
        mybox->integrate_u_fixed_r(_d_cDensity + j*M, _d_rhou + j*Nu_tot*M);

        if ( doSmear ) {
            thrust::copy(d_cDensity.begin() + j*M, d_cDensity.begin() + (j+1)*M, d_tmp.begin());
            mybox->convolveTComplexDouble(d_tmp, d_tmp, d_smearFunc);
            thrust::copy(d_tmp.begin(), d_tmp.end(), d_density.begin() + j*M);

            for ( int nt=0 ; nt<Ntens ; nt++ ) {
                const int off = (j*Ntens + nt) * M;
                thrust::copy(d_cStilde.begin() + off, d_cStilde.begin() + off + M, d_tmp.begin());
                mybox->convolveTComplexDouble(d_tmp, d_tmp, d_smearFunc);
                thrust::copy(d_tmp.begin(), d_tmp.end(), d_Stilde.begin() + off);
            }
        }

        // Not smearing, so just equate the fields
        else {
            thrust::copy(d_cDensity.begin() + j*M, d_cDensity.begin() + (j+1)*M, 
                d_density.begin() + j*M);
            thrust::copy(d_cStilde.begin() + j*Ntens*M, d_cStilde.begin() + (j+1)*Ntens*M, 
                d_Stilde.begin() + j*Ntens*M);
        }

        // Accumulate onto the species
        thrust::transform(d_density.begin() + j*M, d_density.begin() + (j+1)*M, 
            sp.d_density.begin(), sp.d_density.begin(), thrust::plus<thrust::complex<double>>());

        if ( sp.tensor_field == 1 ) {
            thrust::transform(d_Stilde.begin() + j*Ntens*M, d_Stilde.begin() + (j+1)*Ntens*M, 
                sp.d_S.begin(), sp.d_S.begin(), thrust::plus<thrust::complex<double>>());

            thrust::transform(d_rhou.begin() + j*Nu_tot*M, d_rhou.begin() + (j+1)*Nu_tot*M, 
                sp.d_density_ru.begin(), sp.d_density_ru.begin(), 
                thrust::plus<thrust::complex<double>>());
        }
    }// j=0:nUnique

}

// Builds the tables of phase shifts psi_alpha(k_alpha, u) = exp(-i b k_alpha u_alpha).
// Multiplying a field's Fourier transform by prod_alpha psi_alpha translates it
// by +b*u: f(r) -> f(r - b u). Stored as psi_alpha[n*Nu + j] for grid index n
// along alpha and orientation j.
void LCMolec::finish_initialization() {
    const int Nu = mybox->Nu_tot;
    const int Dim = mybox->returnDimension();
    const std::complex<double> I(0.0, 1.0);

    if ( int(mybox->uvecs.size()) != Dim*Nu ) {
        die("orientation grid not initialized before LCMolec::finish_initialization");
    }

    thrust::host_vector<thrust::complex<double>>* psi[3] = {&psi_x, &psi_y, &psi_z};

    for ( int a=0 ; a<Dim ; a++ ) {
        const int Na = mybox->Nx[a];
        psi[a]->resize(Nu*Na);

        for ( int n=0 ; n<Na ; n++ ) {
            // Wavevector along this direction, same ordering as Box::get_kD
            double k = 2.0 * M_PI * double( 2*n < Na ? n : n - Na ) / double(mybox->L[a]);

            for ( int j=0 ; j<Nu ; j++ ) {
                double u = mybox->uvecs[Dim*j + a];

                // The Nyquist mode of an even grid has no well-defined
                // translation; drop it so translated real fields stay real
                if ( Na % 2 == 0 && n == Na/2 ) {
                    (*psi[a])[n*Nu + j] = 0.0;
                }
                else {
                    (*psi[a])[n*Nu + j] = exp(-I * bond_length * k * u);
                }
            }// j=0:Nu
        }
    }

    d_psi_x = psi_x;
    d_psi_y = psi_y;
    if ( Dim == 3 ) d_psi_z = psi_z;

    std::cout << "    psi_alpha variables initialized for LCMolec" << std::endl;
}


// Recomputes number of molecules for GCE simulations
void LCMolec::recomputeNmolecs() {
    if ( activity > 0.0 ) {
        nmolecs = mybox->C * phi * mybox->Vfree * mybox->Nr / double(Ntot);   
    }
}

// No linear (semi-implicit) coefficients are defined for the rigid mesogen yet;
// this only sets the number of molecules from the volume fraction.
void LCMolec::computeLinearTerms() {
    if ( phi > 0.0 ) {
        nmolecs = mybox->C * phi * mybox->Vfree * mybox->Nr / double(Ntot);
        std::cout << " LC nSites = " << nmolecs * double(Ntot) << " nmolecs = " << nmolecs << std::endl;
    }
}
