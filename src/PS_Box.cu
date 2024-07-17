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
    std::cout << "In read input!" << std::endl;

    // Set some preliminary/default variables
    M = 1;
    V = 1.0;
    blockSize = 512;
    MAXBONDS = 2;
    MAXANGLES = 3;
    Nr = 100;
    rho0 = C = -1.0;
    nstot = nBondsTot = nAnglesTot = 0;

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
            // std::cout << "Line read: " << line << std::endl;

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

            else if ( firstWord == "logFreq" || firstWord == "logfreq" ) { 
                iss >> logFreq; 
            }

            else if ( firstWord == "molecule" ) {
                std::string nextWord;
                iss >> nextWord;
                if ( nextWord == "linear" ) {
                    makeLinear(iss);
                }
            }

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

            else if ( firstWord == "species" ) {
                species.push_back(PS_Species(iss, this));
            }

            else {
                std::string s1 = "Invalid keyword " + firstWord + " in FTS_Box::readInput()";
                die(s1.c_str());
            }
            std::cout << "Finished input line: " << line << std::endl;

        }// while ( iss >> firstWord && firstWord != "endBox" ) 
        
        
        if ( firstWord == "endBox" ) {
            std::cout << "endBox caught in outer loop" << std::endl;
            break;
        }

    }// while (!inp.eof())

    if ( nstot == 0 ) {
        die("Box created with no particles?!?");
    }

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

    // gpuGrid, block sizes
    M_Block = blockSize;
    M_Grid = (int)ceil((double)(M) / M_Block);

    // Define C, rho0 depending on what is given
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


    // Count number of bonds, angles
    for ( int i=0 ; i<nstot ; i++ ) {
        for ( int j=0 ; j<nBonds[i] ; j++ ) {
            if ( bondedTo[i*MAXBONDS+j] > i ) nBondsTot++;
        }

        for ( int j=0 ; j<nAngles[i] ; j++ ) {
            die("angle counter not set up in PS_Box.cu::readInput");
        }
    }

    writeDataConfig("init.input.data");
    std::cout << "Initial config in data file format written to init.input.data" << std::endl;

    simTime = time(0);

}// End of readInput()



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


// Finds the index in the species vector with the label 'testLabel'
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

void PS_Box::writeDataConfig(std::string filename) {
    
    std::ofstream out(filename);

    out << "Created by MATILDA.FT\n\n" ;

    out << nstot << " atoms" << std::endl;
    out << nBondsTot << " bonds" << std::endl;
    out << nAnglesTot << " angles" << std::endl;

    // blank line
    out << std::endl;

    // Find number of particle species
    int max = -342332;
    for ( int i=0 ; i<nstot ; i++ ) {
        if ( intSpecies[i] > max ) max = intSpecies[i];
    }
    out << max+1 << " atom types" << std::endl;

    // Find number of bond types
    max = -1;
    for ( int i=0 ; i<nstot ; i++ ) {
        for ( int j=0 ; j<nBonds[i]; j++ ) 
            if ( bondType[i*MAXBONDS+j] > max ) max = bondType[i*MAXBONDS+j];
    }
    out << max+1 << " bond types" << std::endl;

    // Find number of angle types
    max = -1;
    for ( int i=0 ; i<nstot ; i++ ) {
        for ( int j=0 ; j<nAngles[i]; j++ ) 
            die("angles not set up in writeDataCofnig");
    }
    out << max+1 << " angle types" << std::endl;


    // blank line
    out << std::endl;

    // Box dimensions
    out << "0.0 " << L[0] << " xlo xhi" << std::endl;
    out << "0.0 " << L[1] << " ylo yhi" << std::endl;
    if ( Dim > 2 ) out << "0.0 " << L[2] << " zlo zhi" << std::endl;
    else out << "0.0 1.0 zlo zhi" << std::endl;



    // blank line
    out << std::endl;

    out << "Masses\n" << std::endl;
    for ( int i=0 ; i<species.size(); i++ ) {
        out << i+1 << " " << species[i].mass << std::endl;
    }


    // blank line
    out << std::endl;


    // Write out particle coordinates and types
    out << "Atoms\n" << std::endl;
    for ( int i=0 ; i<nstot; i++ ) {

        out << i+1 << " " << mID[i]+1 << " " << intSpecies[i]+1 << "  " ;
        for ( int j=0 ; j<Dim ; j++ ) 
            out << x[i*Dim+j] << " " ;
        if ( Dim < 3 )
            out << "0.0 " ;
        out << std::endl;

    }// i=0 ; i<nstot


    // blank line
    out << std::endl;


    // Write out bond list and types
    out << "Bonds\n" << std::endl;
    int bondCounter = 0;
    for ( int i=0 ; i<nstot ; i++ ) {

        for ( int j=0 ; j<nBonds[i]; j++ ) {
            // Only write each bond once (when i < bondedTo)
            if ( i < bondedTo[i*MAXBONDS+j] ) {
                out << bondCounter+1 << " " << bondType[i*MAXBONDS+j]+1 << " " << i+1 << " " << bondedTo[i*MAXBONDS+j]+1 << std::endl;
                
                bondCounter++;
            }
        }
    }


    out.close();
}// end of writeDataConfig


// Reallocates all of the 'particle-size' arrays to the new value of nstot, 'newns'.
// This can also be used for the intial allocation 
// ONLY AFFECTS HOST ARRAYS
void PS_Box::allocHostParticleArrays(int newns) {
    x.resize(newns*Dim);
    v.resize(newns*Dim);
    v.resize(newns*Dim);

    species.resize(newns);
    intSpecies.resize(newns);

    mID.resize(newns);

    nBonds.resize(newns);
    bondedTo.resize(newns*MAXBONDS);
    bondType.resize(newns*MAXBONDS);

    nAngles.resize(newns);

}

// Generate a new linear polymer of arbitrary blockiness and add it to the box
void PS_Box::makeLinear(std::istringstream& iss ) {
    if ( rho0 < 0.0 ) die("rho0 must be defined before molecules created!");
    
    int maxBonds = 2;   // Linear polymer specific max of two bonds per site
    int maxAngles = 0;  // Needs to be changed to 3 after angles implemented

    int numBlocks, Ntot = 0;

    // Both set to negative values to determine which keyword given
    double phi = -1.0;  
    int nmolecs = -1; 

    std::string s1;

    // read either phi or nmolecs keyword
    iss >> s1;
    if ( s1 == "phi" ) {
        iss >> phi;
    }
    else if ( s1 == "nmolecs" ) {
        iss >> nmolecs;
    }
    else { 
        std::cout << "invalid keyword: " << s1 << std::endl;
        die("invalid keyword after molecule linear. must be 'phi' or 'nmolecs'"); 
    }

    // Read in the number of blocks
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
    // if volume fraction was read
    if ( phi > 0.0 ) { nmolecs = int( rho0 * V * phi / float(Ntot) ); }

    std::cout << "Generating " << nmolecs << " molecules each with " << Ntot << " sites" << std::endl;

    // particle index to be incremented as particles added
    int ind = nstot;


    // Find starting molecule ID
    // If there were no existing molecules, this logic should start at 0
    int molecInd = -1;
    for ( int i=0 ; i<nstot ; i++ ) {
        if (mID[i] > molecInd) molecInd = mID[i];
    }
    molecInd++;



    // Update number of sites in the box
    nstot += nmolecs * Ntot;
    allocHostParticleArrays(nstot);
    std::cout << "nstot changed values to: " << nstot << std::endl;


    // Main loop over molecules, blocks, sites on each block
    for ( int i=0 ; i<nmolecs ; i++ ) {
        for ( int j=0 ; j<numBlocks; j++ ) {
            
            int speciesVal = findSpeciesInteger(speciesBlocks[j]);
            
            for ( int s=0 ; s<Nblocks[j]; s++ ) {

                // Track species info
                speciesType[ind] = speciesBlocks[j];
                intSpecies[ind] = speciesVal;
                
                
                // Is this a chain end? 
                // If so, place randomly in the box
                if ( j==0 && s==0 ) {
                    for ( int a=0 ; a<Dim ; a++ ) {
                        x[ind*Dim+a] = ran2() * L[a];
                    }
                }

                // Not a chain end: place monomer a unit vector away
                // from previous monomer
                else {
                    double ru[3];
                    random_unit_vec(ru, Dim);

                    for ( int a=0 ; a<Dim ; a++ ) {
                        int prevXInd = (ind-1)*Dim+a;
                        int Xind = ind*Dim+a;

                        x[Xind] = x[prevXInd] + ru[a];

                        if ( x[Xind] > L[a] ) x[Xind] -= L[a];
                        else if ( x[Xind] < 0.0 ) x[Xind] += L[a];
                    }
                }

                // Initialize velocities, forces to 0.0
                for ( int a=0 ; a<Dim ; a++ ) {
                    v[ind*Dim+a] = f[ind*Dim+a] = 0.0;
                }

                nBonds[ind] = 0;
                
                // Initialize bonds
                // If not the first monomer on a chain, make a bond to previous monomer
                if ( j != 0 || s != 0 ) {
                    int nb = nBonds[ind];
                    
                    bondedTo[ind*MAXBONDS+nb] = ind-1;
                    bondType[ind*MAXBONDS+nb] = blockBondType[j];
                    nBonds[ind]++;
                }

                // if not the last monomer on a chain, make a bond to next monomer
                if ( j != (numBlocks-1) || s != (Nblocks[j]-1 ) ) {
                    int nb = nBonds[ind];
                    
                    bondedTo[ind*MAXBONDS+nb] = ind+1;
                    bondType[ind*MAXBONDS+nb] = blockBondType[j];
                    nBonds[ind]++;
                }

                // if ( ind < 35 ) {
                //     std::cout << partic[ind].nBonds << " " << partic[ind].bondedTo[0] 
                // }

                mID[ind] = molecInd;

                // Increment the particle index
                ind++;


            }// s=0:N[j]
        }// j=0:numBlocks; 

        // Increment molecule index
        molecInd++;
    }// i=0:nmolecs


    std::cout << "nstot is " << nstot << " after molecule creation" << std::endl;
}
