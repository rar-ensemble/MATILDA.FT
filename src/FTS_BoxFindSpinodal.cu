// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "FTS_Box.h"

#include "fts_molecule.h"
#include "fts_species.h"
#include "fts_potential.h"
#include "include_libs.h"


// Once spinodal point is found, return to spinodal
// while (spinodal phi change > tol) 
//  sweep \mu VT until magnitude of derivative drops
//  save three points along way, once drop occurs, oldest of three is last stable
//  restore state to that stable point
//  save activity, composition data
//  Change activity scaling factor f:
//  f = exp( log(f) * 0.5 )
//  activity *= f
//  resume sweep, saving only last stable point
// 
// NOTE: hard-coding use of three stored points
void FTS_Box::findSpinodal(std::istringstream& iss) {
    
    // Set default parameters
    double phiTol = 1.0E-4;
    double f = 0.8;
    double fScaleFactor = 0.5;      // the 0.5 in the above description
    int molecIndex = 0;
    int nSteps = 10000;
    std::string fileName = "find_spinodal.dat";

    // Parse arguments
    while (iss.tellg() != -1) {

        std::string word;
        iss >> word;

        if ( word == "phiTolerance" ) {
            iss >> phiTol;
        }

        else if ( word == "fScale" ) {
            iss >> fScaleFactor;
        }

        else if ( word == "datFile" ) {
            iss >> fileName;
        }

        else if ( word == "molecule" ) {
            iss >> molecIndex;
            molecIndex -= 1;  // Shift to use zero indexing
        }

        else if ( word == "nSteps" ) {
            iss >> nSteps;
        }

        else if ( word == "initial_f" ) {
            iss >> f;
        }
    }// iss.tellg() != -1

    std::cout << "Searching for spinodal point using: " << std::endl;
    std::cout << "  phiTol = " << phiTol << std::endl;
    std::cout << "  f = " << f << std::endl;
    std::cout << "  fScaleFactor = " << fScaleFactor << std::endl;
    std::cout << "  molecule = " << molecIndex + 1<< std::endl;
    std::cout << "  initalf = " << f << std::endl;

    
    // Bookkeeping variables - generally fixed
    int nSaved = 0;
    int nBkps = 3;
    int nPotentials = this->Potentials.size();
    int nMolecs = this->Molecs.size();
    std::ofstream otp(fileName);

    double *Zstorage, *nStorage;
    Zstorage = (double*) malloc( nMolecs * nBkps * sizeof(double));
    nStorage = (double*) malloc( nMolecs * nBkps * sizeof(double));
    



    // Storage variables
    std::complex<double> *wplBkps, *wmiBkps;

    wplBkps = (std::complex<double>* ) malloc(nPotentials * M * nBkps * sizeof(std::complex<double>));
    wmiBkps = (std::complex<double>* ) malloc(nPotentials * M * nBkps * sizeof(std::complex<double>));
    
    double phiSpinChange = 1.0E2;   // change in spinodal estimate, compared to tolerance
    double oldPhiSpin = 2.0;    // previous spinodal estimate
    double phi[3];
    
    // Perform the initial calculation to find starting composition
    NVT(nSteps);

    // Make initial save of the potentials
    storePotentials(wmiBkps, wplBkps, Zstorage, nStorage, nSaved);
    nSaved++;

    int tryCt = 0;
    
    while ( phiSpinChange > phiTol ) {
        std::string command = "molecule " + std::to_string(molecIndex+1) + " activity scale " \
                              + std::to_string(f);


        int foundSpinodalPoint = 0;
        int initializing = 1;

        // Do the search for a spinodal point:
        // Find case where | d(phi)/d(z) | decreases
        while ( !foundSpinodalPoint ) {
            
            std::istringstream modss(command);

            // Change the activity
            modifyBox(modss);

            // Run new uVT calc
            NVT(nSteps);

            // Still generating the initial three points
            if ( initializing ) {
                std::cout << "  nsaved < 2" << std::endl;
                storePotentials(wmiBkps, wplBkps, Zstorage, nStorage, nSaved);
                nSaved++;
                if ( nSaved == 3 ) {
                    std::cout << "INITIALIZING DONE\n\n" << std::endl;
                    initializing = 0;
                    nSaved = 2;
                }
            }

            // in the 'steady state' and have generated and analyzed three points
            else {
                std::cout << "  nsaved == 2" << std::endl;
                // Move storage down an index
                if (tryCt >= 2 ) shuffleStorage(wmiBkps, wplBkps, Zstorage, nStorage);

                // Store new potentials in open slot
                storePotentials(wmiBkps, wplBkps, Zstorage, nStorage, nSaved);

                // Get indices for tracked molecule
                int m0 = 0 * nMolecs + molecIndex;
                int m1 = 1 * nMolecs + molecIndex;
                int m2 = 2 * nMolecs + molecIndex;

                // Compute normalization for \phi
                double nTot[3] = {0.0, 0.0, 0.0};
                for (int m=0; m<nMolecs ; m++ ) {
                    nTot[0] += nStorage[0*nMolecs+m];
                    nTot[1] += nStorage[1*nMolecs+m];
                    nTot[2] += nStorage[2*nMolecs+m];
                }

                // Compute volume fraction of relevant component
                for ( int i=0 ; i<3; i++ ) {
                    phi[i] = nStorage[i*nMolecs + molecIndex] / nTot[i];
                }

                double dphi01 = fabs( (phi[1] - phi[0]) / (Zstorage[m1] - Zstorage[m0]) );
                double dphi12 = fabs( (phi[2] - phi[1]) / (Zstorage[m2] - Zstorage[m1]) );

                std::cout << "  nSaved: " << nSaved << " nTot: " << nTot[0] << " " << nTot[1] << " " << nTot[2] << std::endl;
                std::cout << "  phi: " << phi[0] << " " << phi[1] << " " << phi[2] << std::endl;
                std::cout << " nStorage:\n";
                for ( int i=0 ; i<nMolecs*3; i++ ) {
                    std::cout << "    " << nStorage[i] << std::endl;
                }


                // Is the derivative decreasing? If so, we found a candidate for the spinodal
                if ( dphi12 < dphi01 / 3.0 ) {
                    foundSpinodalPoint = 1;
                }

                else {
                    std::cout << "  dphi01: " << dphi01 << " dphi12: " << dphi12 << std::endl;
                }

            } // nSaved == 2


            std::cout << "Try " << tryCt << ", " << nSaved << ": ";
            for ( int i=0 ; i<3*nMolecs ; i++ ) {
                std::cout << Zstorage[i] << " " << nStorage[i] << "   " ;
            }
            std::cout << std::endl;

            for ( int m=0 ; m<nMolecs ; m++ ) {
                otp << Molecs[m]->activity << " " << Molecs[m]->nSites << " " ;
            }
            otp << std::endl;

    
            tryCt++;
            // if ( tryCt > 25 ) break;
        }// Not yet found spinodal point
        
        
        std::cout << "spinodal found!\nz = " << Zstorage[0*nMolecs+molecIndex] 
            << " " << Zstorage[1*nMolecs+molecIndex] 
            << " " << Zstorage[2*nMolecs+molecIndex] << std::endl;
        std::cout << "phi = " << phi[0] << " " << phi[1] << " " << phi[2] << std::endl;

        
        std::cout << "\n\n\n\n" << std::endl;
        break;

        phiSpinChange = fabs(phi[0] - oldPhiSpin);
        oldPhiSpin = phi[0];
        
        phi[0] = phi[1] = phi[2] = 0.0;

        f = exp( log(f) * fScaleFactor);
        restorePotentials(wmiBkps, wplBkps, Zstorage, nStorage, 0);
        
        // divides by f to cancel out the next iteration in the loop
        Molecs[molecIndex]->activity = Molecs[molecIndex]->activity / f;

        nSaved = 0;
        std::cout << "\nf changed to: " << f << ", phi_spin \approx " << oldPhiSpin << std::endl;

        // Perform the initial calculation to find starting composition
        NVT(nSteps);

        // Make initial save of the potentials
        storePotentials(wmiBkps, wplBkps, Zstorage, nStorage, nSaved);
        nSaved++;

    } // while phiChange > phiTol

    otp.close();   
}


// Shuffles storage so storage slot 2 can be overwritten
// Current slot 1 --> slot 0
// current slot 2 --> slot 1
void FTS_Box::shuffleStorage(
    std::complex<double>* miBkp,    // [M*nPotentials*3] store array for wmi arrays
    std::complex<double>* plBkp,    // [M*nPotentials*3] store array for wpl arrays
    double* zStore,                 // [3*nMolecs] storage for activity
    double* nStore                  // [3*nMolecs] storage for nmolecules
){
    int nPotentials = this->Potentials.size();


    for ( int p=0 ; p < nPotentials ; p++ ) {
        
        int p0off = 0 * nPotentials * M + p * M;    // offset for slot 0
        int p1off = 1 * nPotentials * M + p * M;    // offset for slot 1
        int p2off = 2 * nPotentials * M + p * M;    // offset for slot 2

        for ( int i=0 ; i<M ; i++ ) {
            miBkp[p0off + i] = miBkp[p1off + i];    // wmi[0] = wmi[1]
            plBkp[p0off + i] = plBkp[p1off + i];    // wpl[0] = wpl[1]

            miBkp[p1off + i] = miBkp[p2off + i];    // wmi[1] = wmi[2]
            plBkp[p1off + i] = plBkp[p2off + i];    // wpl[1] = wpl[2]            
        }
    }

    int nMolecs = this->Molecs.size();

    for (int m=0 ; m<nMolecs ; m++ ) {
        int m0ind = 0 * nMolecs + m;
        int m1ind = 1 * nMolecs + m;
        int m2ind = 2 * nMolecs + m;
        
        zStore[m0ind] = zStore[m1ind];
        zStore[m1ind] = zStore[m2ind];
        
        nStore[m0ind] = nStore[m1ind];
        nStore[m1ind] = nStore[m2ind];
    }
}




// Extracts wmi, wpl from the potentials and stores them in host memory
void FTS_Box::storePotentials(
    std::complex<double>* miBkp,    // [M*nPotentials*3] store array for wmi arrays
    std::complex<double>* plBkp,    // [M*nPotentials*3] store array for wpl arrays
    double* zStore,                 // [3*nMolecs] storage for activity
    double* nStore,                 // [3*nMolecs] storage for nmolecules
    const int ind                   // storage index
) {

    if ( ind >= 3 ) die("Invalid index in storePotentials");

    int nPotentials = this->Potentials.size();

    for ( int p=0 ; p<nPotentials; p++ ) {

        int doWmi = Potentials[p]->wmiAllocated();
        int doWpl = Potentials[p]->wplAllocated();

        // The potentials are thrust arrays
        // Copy them to the host
        if ( doWmi ) Potentials[p]->wmi = Potentials[p]->d_wmi;
        if ( doWpl ) Potentials[p]->wpl = Potentials[p]->d_wpl;

        // Offset for the potential indices
        int pOff = ind * nPotentials * M + p * M;
        for ( int i=0 ; i<M ; i++ ) {
            if ( doWmi ) miBkp[pOff + i] = Potentials[p]->wmi[i];
            if ( doWpl ) plBkp[pOff + i] = Potentials[p]->wpl[i];
        }
    }// p=0:nPotentials

    int nMolecs = this->Molecs.size();

    for (int m=0 ; m<nMolecs ; m++ ) {
        int mInd = ind * nMolecs + m;
        zStore[mInd] = Molecs[m]->activity;
        nStore[mInd] = Molecs[m]->nSites;
    }

    std::cout << "  FINISHED STORE POTENTIALS WITH nSaved = " << ind << std::endl;

}


// Extracts wmi, wpl from storage in host memory and restores them to potentials
void FTS_Box::restorePotentials(
    std::complex<double>* miBkp,    // [M*nPotentials*3] store array for wmi arrays
    std::complex<double>* plBkp,    // [M*nPotentials*3] store array for wpl arrays
    double* zStore,                 // [3*nMolecs] storage for activity
    double* nStore,                 // [3*nMolecs] storage for nmolecules
    const int ind) {

    if ( ind >= 3 ) die("Invalid index in storePotentials");

    int nPotentials = this->Potentials.size();

    for ( int p=0 ; p<nPotentials; p++ ) {
        int doWmi = Potentials[p]->wmiAllocated();
        int doWpl = Potentials[p]->wplAllocated();

        // Offset for the potential indices
        int pOff = ind * nPotentials * M + p * M;
        for ( int i=0 ; i<M ; i++ ) {
            if ( doWmi ) Potentials[p]->wmi[i] = miBkp[pOff + i];
            if ( doWpl ) Potentials[p]->wpl[i] = plBkp[pOff + i];
        }

        
        // The potentials are thrust arrays
        // Copy them to the device
        if ( doWmi ) Potentials[p]->d_wmi = Potentials[p]->wmi;
        if ( doWpl ) Potentials[p]->d_wpl = Potentials[p]->wpl;
    }
    


    int nMolecs = this->Molecs.size();

    for (int m=0 ; m<nMolecs ; m++ ) {
        int mInd = ind * nMolecs + m;
        Molecs[m]->activity = zStore[mInd];
        std::cout << "  Molec " << m << " activity restored to: " << Molecs[m]->activity << std::endl;
    }
}