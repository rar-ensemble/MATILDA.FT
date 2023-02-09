// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "fts_molecule_linear.h"
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/complex.h>
#include <thrust/copy.h>
#include "FTS_Box.h"
#include "fts_species.h"

LinearMolec::~LinearMolec(){}

LinearMolec::LinearMolec(std::istringstream& iss, FTS_Box* p_box) : FTS_Molec(iss, p_box) {
    // iss comes into this routine having already passed "molecule" and "linear"
    // structure of iss should be:
    // numBlocks, integer
    // Ns for block 1, type for block 1. Type is a char, should be one of the species types
    // Ns for block 2, type for block 2...
    // Repeat for number of blocks


    iss >> numBlocks;

    Ntot = 0;
    blockSpecies.resize(numBlocks);
    intSpecies.resize(numBlocks);
    d_intSpecies.resize(numBlocks);
    d_N.resize(numBlocks);
    N.resize(numBlocks);

    // Resize density arrays for multiblocks
    density.resize(numBlocks*mybox->M);
    d_density.resize(numBlocks*mybox->M);
    d_cDensity.resize(numBlocks*mybox->M);

    for ( int j=0 ; j<numBlocks; j++ ) {
        iss >> N[j];
        iss >> blockSpecies[j];

        Ntot += N[j];
    }

    // Copy block lengths to device
    d_N = N;

    // Allocate forward and backwards propagators
    d_q.resize(Ntot*mybox->M);
    d_qdag.resize(Ntot*mybox->M);


    // Check for symmetry of the molecule. If symmetric, d_qdag = d_q
    // saving Ntot*M calcs.
    for ( int j=0 ; j<numBlocks/2 ; j++ ) {
        if ( ( blockSpecies[j] != blockSpecies[numBlocks-1-j] ) ||    // Block Species are unequal
             (N[j] != N[numBlocks-1-j]) )  {                    // Block lengths are unequal
             
            isSymmetric = false;
            break;
        }
    }
    std::cout << "Is molecule symmetric: " << isSymmetric << std::endl;

    // Determine integer block Species
    for ( int j=0 ; j<numBlocks; j++ ) {
        for ( int i=0 ; i<mybox->Species.size(); i++ ) {
            if ( blockSpecies[j] == mybox->Species[i].fts_species ) {
                intSpecies[j] = i;
            }
        }
    }

    d_intSpecies = intSpecies;



    // Initialize bond potential
    thrust::device_vector<thrust::complex<double>> bond_fft(mybox->M);

    double k2, *kv;
    kv = new double[mybox->returnDimension()];
    for ( int i=0 ; i<mybox->M ; i++ ) {
        k2 = mybox->get_kD(i, kv);

        //bond_fft[i] = exp(-k2/6.0);
        bond_fft[i] = exp(-k2/(mybox->Nr-1.0));
    }
    delete kv;

    // Send bond potential to device
    d_bond_fft = bond_fft;


}

struct NegExponential {
    __host__ __device__
        thrust::complex<double> operator()(const thrust::complex<double> &z) {
            return exp(-z);
    }
};

void LinearMolec::calcPropagators() {
    
    //thrust::device_vector<thrust::complex<double>> &W = mybox->Species[intSpecies[0]].d_w;
    thrust::device_vector<thrust::complex<double>> W(mybox->M);
    // W = mybox->Species[intSpecies[0]].d_w;

    thrust::device_vector<thrust::complex<double>> qf(mybox->M);
    thrust::device_vector<thrust::complex<double>> hf(mybox->M);
    thrust::device_vector<thrust::complex<double>> a(mybox->M);
    thrust::device_vector<thrust::complex<double>> expW(mybox->M);

    // To normalize FTs
    thrust::device_vector<thrust::complex<double>> norm(mybox->M);
    thrust::fill(norm.begin(), norm.end(), 1.0/double(mybox->M));

    //////////////////////////////////
    // Calculate forward propagator //
    //////////////////////////////////
    int ind=0;

    for ( int b=0 ; b<numBlocks ; b++ ) {
        W = mybox->Species[intSpecies[b]].d_w;

    
        // expW = exp(-W)
        thrust::transform(W.begin(), W.end(), expW.begin(), NegExponential()); 



        for ( int s=0; s<N[b] ; s++ ) {
            
            // Initial condition
            if ( ind==0 ) {
                // qdag = exp(W)
                thrust::copy(expW.begin(), expW.end(), d_q.begin());
                ind += 1;              
            }

            // Rest of the chain
            else {

                thrust::copy(d_q.begin()+(ind-1)*mybox->M, d_q.begin()+ind*mybox->M, qf.begin());

                // a = FT[ q[ind-1] ]
                mybox->cufftWrapperDouble(qf, a, 1);            

                // Multiply bond_fft with FT[q(ind-1)]
                // h = a * d_bond_fft
                thrust::transform(d_bond_fft.begin(), d_bond_fft.end(), a.begin(), hf.begin(), 
                    thrust::multiplies<thrust::complex<double>>());

                // qf = IFT[h]
                mybox->cufftWrapperDouble(hf, qf, -1);

                // d_q = qf * expW
                thrust::transform(qf.begin(), qf.end(), expW.begin(), d_q.begin()+ind*mybox->M,
                    thrust::multiplies<thrust::complex<double>>());

                ind += 1;
            } // ind != 0


        } // s=0:N[b]
    }// b=0:numBlocks


    // If the molecule is symmetric, d_qdag = d_q
    if ( isSymmetric ) {
        d_qdag = d_q;
    }

    //////////////////////////////
    // Complimentary propagator //
    //////////////////////////////
    else {
        ind=0;

        for ( int b=0 ; b<numBlocks ; b++ ) {
            W = mybox->Species[intSpecies[numBlocks-b-1]].d_w;
            
            thrust::transform(W.begin(), W.end(), expW.begin(), NegExponential()); 


            for ( int s=0; s<N[numBlocks-b-1] ; s++ ) {
                
                // Initial condition
                if ( ind==0 ) {
                    // qdag = exp(W)
                    thrust::copy(expW.begin(), expW.end(), d_qdag.begin());
                    ind += 1;
                }

                // Rest of the chain
                else {
                    
                    // qf = qdag[ind-1][:]
                    thrust::copy(d_qdag.begin()+(ind-1)*mybox->M, d_qdag.begin()+ind*mybox->M, qf.begin());

                    // a = FT[ qdag[ind-1] ]
                    mybox->cufftWrapperDouble(qf, a, 1);

                    // Multiply bond_fft with FT[q(ind-1)]
                    // h = a * d_bond_fft
                    thrust::transform(d_bond_fft.begin(), d_bond_fft.end(), a.begin(), hf.begin(), 
                        thrust::multiplies<thrust::complex<double>>());

                    // qf = IFT[h]
                    mybox->cufftWrapperDouble(hf, qf, -1);

                    // d_q = qf * expW
                    thrust::transform(qf.begin(), qf.end(), expW.begin(), d_qdag.begin()+ind*mybox->M,
                        thrust::multiplies<thrust::complex<double>>());

                    ind += 1;
                }

            } // s=0:N[numBlocks-b-1]
        }// b=0:numBlocks
    }// Calculating complimentary propagator

    // debug
    if ( intSpecies[0] == 1 ) {
        thrust::host_vector<thrust::complex<double>> htmp(mybox->M);
        thrust::copy(d_q.begin()+0, d_q.begin()+mybox->M, htmp.begin());
        mybox->writeTComplexGridData("q0.dat", htmp);

        thrust::copy(d_q.begin()+mybox->M, d_q.begin()+2*mybox->M, htmp.begin());
        mybox->writeTComplexGridData("q1.dat", htmp);

        thrust::host_vector<thrust::complex<double>> hW(mybox->M);
        hW = W;
        mybox->writeTComplexGridData("wField.dat", hW);
    }

    // Calculate the partition function
    this->Q = thrust::reduce(d_q.begin()+(Ntot-1)*mybox->M, d_q.begin()+Ntot*mybox->M, thrust::complex<double>(0.0), 
                        thrust::plus<thrust::complex<double>>()) * mybox->gvol / mybox->V;


    // debug
    std::cout << "species " << intSpecies[0] << " Q: " << this->Q << std::endl;

}

struct Exponential {
    __host__ __device__
        thrust::complex<double> operator()(const thrust::complex<double> &z) {
            return exp(z);
    }
};

void LinearMolec::calcDensity() {

    // First, calculate the propagators
    calcPropagators();


    int M = mybox->M;
    thrust::complex<double> factor = nmolecs / Q / mybox->V;
    std::cout << "species " << intSpecies[0] << " factor: " << factor << " nK: " << nmolecs << std::endl;

    thrust::device_vector<thrust::complex<double>> W(mybox->M);
    thrust::device_vector<thrust::complex<double>> q_qdag(mybox->M);
    thrust::device_vector<thrust::complex<double>> temp(mybox->M);
    thrust::device_vector<thrust::complex<double>> expW(mybox->M);
    // Zero all block density fields
    thrust::fill(d_cDensity.begin(), d_cDensity.end(), 0.0);
    int ind = 0;
    for ( int b=0 ; b<numBlocks; b++ ) {
        

        // Accumulate density of all monomers in this block
        for ( int s=0 ; s<N[b]; s++ ) {

            // q_qdag[:] = q[ind][:] * qdag[Ntot-1-ind][:]
            thrust::transform(d_q.begin()+ind*M, d_q.begin()+(ind+1)*M, 
                                d_qdag.begin()+(Ntot-1-ind)*M, q_qdag.begin(),
                                thrust::multiplies<thrust::complex<double>>());                                 

            // cDensity[:] += q_qdag[:]
            thrust::transform(q_qdag.begin(), q_qdag.end(), d_cDensity.begin()+b*M, d_cDensity.begin()+b*M,
                                thrust::plus<thrust::complex<double>>());

            ind += 1;
        }

        W = mybox->Species[intSpecies[b]].d_w;
        // expW = exp(W)
        thrust::transform(W.begin(), W.end(), expW.begin(), Exponential()); 
        
        // factor = n / Q / V
        thrust::fill(temp.begin(), temp.end(), factor);

        // temp <- factor * expW
        thrust::transform(temp.begin(), temp.end(), expW.begin(), temp.begin(), 
                            thrust::multiplies<thrust::complex<double>>()); 

        // Multiply this blocks density by expW*factor
        thrust::transform(d_cDensity.begin()+b*M, d_cDensity.begin()+(b+1)*M, temp.begin(), 
                            d_cDensity.begin()+b*M, thrust::multiplies<thrust::complex<double>>()); 

    }

    // Define total density as juts center density for now. needs to be convolved
    // with shape functions once those are implemented.
    d_density = d_cDensity;

    thrust::host_vector<thrust::complex<double>> htmp(mybox->M);
    htmp = d_density;
    if ( intSpecies[0] == 0 ) mybox->writeTComplexGridData("rho0molec.dat", htmp);
    else mybox->writeTComplexGridData("rho1molec.dat", htmp);
    

    // Finally, accumulate density onto the relevant species field
    for ( int b=0 ; b<numBlocks ; b++ ) {
        int is = intSpecies[b];
        thrust::transform(d_density.begin()+b*M, d_density.begin()+(b+1)*M, mybox->Species[is].d_density.begin(), 
                            mybox->Species[is].d_density.begin(), thrust::plus<thrust::complex<double>>());
    }

}


// Once smearing is implemented, smear functions need to 
// be included in the linear coefficients
void LinearMolec::computeLinearTerms() {
    nmolecs = mybox->C * phi * mybox->V * mybox->Nr / (double(Ntot));

    double alpha = double(Ntot) / double(mybox->Nr);

    // n*Ntot**2 / (Nr * V)
    double prefactor = nmolecs * alpha * double(Ntot) / mybox->V;
    thrust::host_vector<thrust::complex<double>> pref(mybox->M, prefactor);
    
    // Declare htmp storage variable
    thrust::host_vector<thrust::complex<double>> htmp(mybox->M);
    thrust::host_vector<thrust::complex<double>> gd(mybox->M);

    // Get the total Debye function for Helfand potential
    mybox->computeHomopolyDebye(gd, alpha);


    // htmp = prefactor * gd
    thrust::transform(pref.begin(), pref.end(), gd.begin(), htmp.begin(),
        thrust::multiplies<thrust::complex<double>>());
    

    // Find the Helfand potential and add this molecules contribution
    for ( int i=0 ; i<mybox->Potentials.size(); i++ ) {
        if ( mybox->Potentials[i]->printStyle() == "Helfand" ) {
            thrust::host_vector<thrust::complex<double>> Atmp(mybox->M);
            Atmp = mybox->Potentials[i]->d_Akpl;

            thrust::transform(htmp.begin(), htmp.end(), Atmp.begin(),
                Atmp.begin(), thrust::plus<thrust::complex<double>>());

            mybox->Potentials[i]->d_Akpl = Atmp;
        }
    }


    // Add to Flory potentials

    // If homopolymer, just add htmp to d_Akpl and be done
    if ( numBlocks == 1 ) {
        for ( int i=0 ; i<mybox->Potentials.size(); i++ ) {
            if ( mybox->Potentials[i]->printStyle() == "Flory" ) {
                thrust::host_vector<thrust::complex<double>> Atmp(mybox->M);
                Atmp = mybox->Potentials[i]->d_Akpl;

                thrust::transform(htmp.begin(), htmp.end(), Atmp.begin(), Atmp.begin(),
                                    thrust::plus<thrust::complex<double>>());

                mybox->Potentials[i]->d_Akpl = Atmp;
            }
        }
    }// numBlocks == 1 (homopolymer)


/*    // Block polymers
    else {
        thrust::host_vector<thrust::complex<double>> gaa(mybox->M);
        thrust::host_vector<thrust::complex<double>> gac(mybox->M);
        thrust::host_vector<thrust::complex<double>> gcc(mybox->M);

        for ( int i=0; i<mybox->Potentials.size() ; i++ ) {
            if ( mybox->Potentials[i]->printStyle() == "Flory" ) {
                for ( int j=0 ; j<numBlocks-1 ; j++ ) {
                    for ( int k=j+1 ; k<numBlocks ; k++ ) {

                    }// k=j+1:numBlocks
                }// j=0:numBlocks
            }// if potentialStyle == "Flory"
        }

    }// numBlocks > 1
*/
}