// Copyright (c) 2023 University of Pennsylvania
// Part of MATILDA.FT, released under the GNU Public License version 2 (GPLv2).


#include "PS_Box.h"
#include "random.h"
#include "include_libs.h"

void die(const char*);
double ran2(void);
void random_unit_vec(double*, int);

// Executes the commands for a given time step
// Updates all fields, then recomputes all molecule densities
// then populating species densities
void PS_Box::doTimeStep(int step) {


    // Write log data
    if ( step % logFreq == 0 ) {
        writeData(step);
    }

}


// Write Hamiltonian terms to output file
void PS_Box::writeData(int step) {

    // computeHamiltonian();

    OTP.open("ps_data.dat", std::ios_base::app);
    std::string outline;

    OTP << step ;

    OTP << std::endl;
    std::cout << std::endl;

    OTP.close();
}


void PS_Box::initializeSim() {
    

    // Initialize the output stream
    OTP.open("ps_data.dat");
    OTP.close();


}


void PS_Box::readInput(std::ifstream& inp) {

    // Set some preliminary/default variables
    M = 1;
    V = 1.0;
    blockSize = 512;
    rho0 = C = -1.0;
    ntot = 0;

    // Some default values
    logFreq = 100;

    std::string line, firstWord;

    bool readDimension = false;

    while (!inp.eof()) {
        getline(inp, line);

        if ( line.length() == 0 || line.at(0) == '#')
            continue;

        std::istringstream iss(line);
        
        while ( iss >> firstWord ) {

            if ( firstWord == "endBox" ) {
                std::cout << "endBox caught" << std::endl;
                break;
            }

            if ( firstWord == "blocksize" || firstWord == "blockSize" ) { iss >> blockSize ; }

            // Commands are alphabetical from here on
            else if ( firstWord == "boxLengths" ) {
                if (!readDimension) 
                    std::cout << "\nWARNING! WARNING!\nboxLengths read before dimension specified, assuming default value of 2"<<std::endl;

                for ( int j=0 ; j<Dim ; j++ ) {
                    iss >> L[j];
                    V *= L[j];
                    if ( Nx[0] > 0 ) { dx[j] = L[j] / double(Nx[j]); }
                }
            }


            else if ( firstWord == "Dim" ) {
                iss >> Dim;
                setDimension(Dim);
                readDimension = true;
            }

            else if ( firstWord == "grid" ) {
                if ( !readDimension ) { die("Dim must be defined before grid!" );}
                for ( int j=0 ; j<Dim ; j++ ) {
                    iss >> Nx[j];
                    M *= Nx[j];
                    if ( L[0] > 0.0 ) { dx[j] = L[j] / double(Nx[j]); }
                }
            }

            else if ( firstWord == "logFreq" || firstWord == "logfreq" ) { iss >> logFreq; }


            else if ( firstWord == "Nr" ) {
                iss >> Nr;
            }

            else if ( firstWord == "Nx") {
                iss >> Nx[0];
                M *= Nx[0];
            }

            else if (firstWord == "Ny") {
                iss >> Nx[1];
                M *= Nx[1];
            }

            else if (firstWord == "Nz") {
                if ( Dim != 3 ) die("Nz supplied in a non-3D simulation!");
                iss >> Nx[2];
                M *= Nx[2];
            }


            else if (firstWord == "randSeed" || firstWord == "RAND_SEED") {
                iss >> idum;
            }

            else if (firstWord == "rho0") {
                iss >> rho0;
            }

            else {
                std::string s1 = "Invalid keyword " + firstWord + " in FTS_Box::readInput()";
                die(s1.c_str());
            }
            std::cout << line << std::endl;

        }// while ( iss >> firstWord && firstWord != "endBox" ) 
        
        
        if ( firstWord == "endBox" ) {
            std::cout << "endBox caught in outer loop" << std::endl;
            break;
        }

    }// while (!inp.eof())

    // After input read, make the FFT plan
    // This currently assumes complex-double to complex-double transforms
    // change Z2Z to C2C to switch to cpx-float
    if ( this->Dim == 2 ) 
        cufftPlan2d(&fftplan, Nx[1], Nx[0], CUFFT_Z2Z);
    if ( this->Dim == 3 ) 
        cufftPlan3d(&fftplan, Nx[2], Nx[1], Nx[0], CUFFT_Z2Z);

    // Define gvol, dx
    gvol = 1.0;
    for ( int j=0 ; j<Dim ; j++ ) {
        dx[j] = L[j] / double(Nx[j]);
        gvol *= dx[j];
    }

    M_Block = blockSize;
    M_Grid = (int)ceil((double)(M) / M_Block);

    if ( rho0 > 0 && C > 0 ) { die("Cannot define both C and rho0!"); }

    double Rg = pow( (Nr-1.0)/6.0, 0.5);
    if ( rho0 > 0 ) {
        C = rho0/Nr;
        for ( int j=0; j<Dim ; j++ ) C *= Rg;
        std::cout << "Using Nr = " << Nr << ", computed C = " << C << std::endl;
    }
    else if ( C > 0 ) {
        rho0 = C * Rg * Rg * Rg / Nr;
        std::cout << "Using Nr = " << Nr << ", computed rho0 = " << rho0 << " [b^-3]" << std::endl;
    }

    simTime = time(0);

}

void PS_Box::writeFields() {
}

void PS_Box::writeTime() {

    int dt = time(0) - simTime;
    std::cout << "Total simulation time: " << dt / 60 << "m" << dt % 60 << "sec" << std::endl;
    
    dt = ftTimer;
    std::cout << "Total FT time: " << dt / 60 << "m" << dt % 60 << "sec" << std::endl;

}



PS_Box::~PS_Box() {}

PS_Box::PS_Box(std::istringstream& iss ) : Box(iss) {
    std::string s1;

    std::cout << "Made PS_Box " << std::endl;
}

int PS_Box::findSpeciesInteger(std::string testLabel) {
    int id = -1;
    for ( int i=0 ; i<species.size() ; i++ ) {
        if ( species[i].isSpecies( testLabel) ) {
            id = i;
            break;
        }
    }
    if ( id < 0 ) die("Species label not found!");

    return id;
}

void PS_Box::makeLinear(std::istringstream& iss ) {
    if ( rho0 < 0.0 ) die("rho0 must be defined before molecules created!");
    
    int numBlocks, Ntot = 0;
    double phi;   

    iss >> phi;
    iss >> numBlocks; 
    
    std::vector<int> Nblocks(numBlocks);
    std::vector<int> blockBondType(numBlocks,0);
    std::vector<int> wlcFlag(numBlocks,0);
    std::vector<int> drudeFlag(numBlocks,0);
    std::vector<int> chargeFlag(numBlocks,0);
    std::vector<std::string> speciesBlocks(numBlocks);
    std::vector<int> intSpeciesBlocks(numBlocks);

    // Store the basic block info
    for (int j=0 ; j<numBlocks; j++ ) {
        iss >> Nblocks[j];
        iss >> speciesBlocks[j];

        Ntot += Nblocks[j];
    }

    // Check for optional arguments
    if ( iss.tellg() != -1 ) {
        std::string s1;
        iss >> s1;
        if ( s1 == "wlc" || s1 == "drude" || s1 == "charge" ) {
            die("worm-like chains, drude oscillators, and charges not implemented!");
        }
        else if ( s1 == "bondType" || s1 == "bondtype" ) {
            die("bond types not set up in ps_box make linear");
            // make sure to decide on how to handle junction cases, document it
        }
    }

    // Compute number of molecules of this type to add
    int nmolecs = int( rho0 * V * phi / float(Ntot) );
    std::cout << "Generating " << nmolecs << " molecules each with " << Ntot << "sites" << std::endl;

    // particle index to be incremented as particles added
    int ind = ntot;

    // Update number of sites in the box
    ntot += nmolecs * Ntot;
    partic.resize(ntot);


    // Main loop over molecules, blocks, sites on each block
    for ( int i=0 ; i<nmolecs ; i++ ) {
        for ( int j=0 ; j<numBlocks; j++ ) {
            
            int speciesVal = findSpeciesInteger(speciesBlocks[j]);
            
            for ( int s=0 ; s<Nblocks[j]; s++ ) {

                // Re-size this particles' dimensions
                partic[ind].setSizes(Dim,2,0);


                // Track species info
                partic[ind].species = speciesBlocks[j];
                partic[ind].intSpecies = speciesVal;
                
                
                // Is this a chain end? 
                // If so, place randomly in the box
                if ( j==0 && s==0 ) {
                    for ( int a=0 ; a<Dim ; a++ ) {
                        partic[ind].x[a] = ran2() * L[a];
                    }
                }

                // Not a chain end: place monomer a unit vector away
                // from previous monomer
                else {
                    double ru[3];
                    random_unit_vec(ru, Dim);

                    for ( int a=0 ; a<Dim ; a++ ) {
                        partic[ind].x[a] = partic[ind-1].x[a] + ru[a];
                    }
                }

                // Initialize velocities, forces to 0.0
                for ( int a=0 ; a<Dim ; a++ ) {
                    partic[ind].v[a] = partic[ind].f[a] = 0.0;
                }


                // Initialize bonds
                // If not the first monomer on a chain, make a bond to previous monomer
                if ( j != 0 || s != 0 ) {
                    int nb = partic[ind].nBonds;
                    
                    partic[ind].bondedTo[nb] = ind-1;
                    partic[ind].bondedTo[nb] = blockBondType[j];
                    partic[ind].nBonds++;
                }

                // if not the last monomer on a chain, make a bond to next monomer
                if ( j != (numBlocks-1) || s != (Nblocks[j]-1 ) ) {
                    int nb = partic[ind].nBonds;

                    partic[ind].bondedTo[nb] = ind+1;
                    partic[ind].bondedTo[nb] = blockBondType[j];
                    partic[ind].nBonds++;
                }

                // Increment the particle index
                ind++;

            }// s=0:N[j]
        }// j=0:numBlocks; 
    }// i=0:nmolecs

}
