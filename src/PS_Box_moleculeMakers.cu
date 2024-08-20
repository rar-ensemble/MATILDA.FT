#include "PS_Box.h"
#include "random.h"
#include "include_libs.h"
#include "gsd.h"
#include <algorithm>
#include <map>

void die(const char*);
double ran2(void);
void random_unit_vec(double*, int);

// Generate a new linear polymer of arbitrary blockiness and add it to the box
void PS_Box::makeLinear(std::istringstream& iss ) {
    if ( rho0 < 0.0 ) die("rho0 must be defined before molecules created!");

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


    // Update number of sites in the box
    nstot += nmolecs * Ntot;
    allocHostParticleArrays(nstot);
    std::cout << "nstot changed values to: " << nstot << ", starting index: " << ind << std::endl;



    // Find starting molecule ID
    int molecInd = -1;
    for ( int i=0 ; i<nstot ; i++ ) {
        if (mID[i] > molecInd) molecInd = mID[i];
    }
    if ( molecInd < 0 ) molecInd = 0;




    // Main loop over molecules, blocks, sites on each block
    for ( int i=0 ; i<nmolecs ; i++ ) {
        for ( int j=0 ; j<numBlocks; j++ ) {
            int speciesVal = findSpeciesInteger(speciesBlocks[j]);
            
            for ( int s=0 ; s<Nblocks[j]; s++ ) {

                // Track species info
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
