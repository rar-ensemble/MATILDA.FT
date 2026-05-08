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
    int angleFlag = 0;

    // Flag for dealing with charges on this molec type
    int doMolecCharges = 0;

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
    std::vector<int> blockBondType(numBlocks,1);
    std::vector<int> blockAngleType(numBlocks,0);
    std::vector<int> drudeFlag(numBlocks,0);
    std::vector<int> chargeFlag(numBlocks,0);
    std::vector<std::string> speciesBlocks(numBlocks);
    std::vector<int> intSpeciesBlocks(numBlocks);
    std::vector<float> blockCharge(numBlocks, 0.0);
    
    std::vector<float> Rmin(Dim,0.0);
    std::vector<float> Rmax(Dim);
    for ( int j=0 ; j<Dim ; j++ ) Rmax[j] = L[j];

    // Store the basic block info
    for (int j=0 ; j<numBlocks; j++ ) {
        iss >> Nblocks[j];
        iss >> speciesBlocks[j];

        Ntot += Nblocks[j];
    }

    // Check for optional arguments
    while ( iss.tellg() != -1 ) {

        iss >> s1;
        if ( s1 == "drude" ) {
            die("drude oscillators not implemented!");
        }

        else if ( s1 == "charge" ) {
            if ( this->doCharges == 0 ) {
                this->enableCharges();
            }

            doMolecCharges = 1;
            
            for ( int j=0 ; j<numBlocks ; j++ ) {
                float q1;
                iss >> q1;
                blockCharge[j] = q1;
            }
        }// s1==charge

        else if ( s1 == "bondType" || s1 == "bondtype" ) {
            for ( int j=0 ; j<numBlocks ; j++ ) {
                int t1; 
                iss >> t1;
                blockBondType[j] = t1;
            }
            // make sure to decide on how to handle junction cases, document it
        }
        else if ( s1 == "angleType" || s1 == "angletype" ) {
            for ( int j=0 ; j<numBlocks ; j++ ) {
                int t1;
                iss >> t1;
                blockAngleType[j] = t1;
                std::cout << "  block ang type: " << j << " " << blockAngleType[j] << std::endl;
                angleFlag = 1;
            }
        }
        else if ( s1 == "xrange" ) {
            iss >> Rmin[0];
            iss >> Rmax[0];
        }
        else if ( s1 == "yrange" ) {
            iss >> Rmin[1];
            iss >> Rmax[1];
        }
        else if ( s1 == "zrange" ) {
            if ( Dim != 3 ) { die("z-range defined for a non-3D simulation!"); }
            iss >> Rmin[2];
            iss >> Rmax[2];
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


    // Initializes angle counter to zero
    int aind = ind;
    for ( int i=0 ; i<nmolecs ; i++ ) {
        for ( int j=0 ; j<numBlocks ; j++ ) {
            for (int s=0; s<Nblocks[j] ; s++ ) {
                nAngles[aind] = 0;
                aind++;
            }
        }
    }
    std::cout << "   MAX ANGLE INDEX: " << 3 * nstot * MAXANGLES << std::endl;






    ///////////////////////////////////////////////////////////
    // Main loop over molecules, blocks, sites on each block //
    ///////////////////////////////////////////////////////////
    for ( int i=0 ; i<nmolecs ; i++ ) {

        // backbone monomer index for molecule i
        int bbIndex = 0;
        double ru[3];
        for ( int j=0 ; j<numBlocks; j++ ) {
            int speciesVal = findSpeciesInteger(speciesBlocks[j]);
            
            for ( int s=0 ; s<Nblocks[j]; s++ ) {

                // Track species info
                intSpecies[ind] = speciesVal;
                
                
                // Is this a chain end? 
                // If so, place randomly in the box
                if ( j==0 && s==0 ) {
                    for ( int a=0 ; a<Dim ; a++ ) {
                        x[ind*Dim+a] = ran2() * (Rmax[a] - Rmin[a]) + Rmin[a];
                    }
                }

                // Not a chain end: place monomer a unit vector away
                // from previous monomer
                else {
                    
                    // if first monomer past the end or there are no angles,
                    // re-generate the bond orientation
                    if ( ( s==1 && j==0 ) || (!angleFlag) )
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


                if ( doMolecCharges ) {
                    charges[ind] = blockCharge[j];
                }

                nBonds[ind] = 0;
                
                // Initialize bonds
                // If not the first monomer on a chain, make a bond to previous monomer
                if ( j != 0 || s != 0 ) {
                    int nb = nBonds[ind];
                    
                    bondedTo[ind*MAXBONDS+nb] = ind-1;
                    bondType[ind*MAXBONDS+nb] = blockBondType[j] - 1;
                    nBonds[ind]++;
                }

                // if not the last monomer on a chain, make a bond to next monomer
                if ( j != (numBlocks-1) || s != (Nblocks[j]-1 ) ) {
                    int nb = nBonds[ind];
                    
                    bondedTo[ind*MAXBONDS+nb] = ind+1;
                    bondType[ind*MAXBONDS+nb] = blockBondType[j] - 1;
                    nBonds[ind]++;
                }


                // Initialize angles
                // logic is different from bonds. If *central* particle is
                // in an angle, then i1 and i3 angle lists are both updated.
                // This ensures angles accounted for when one site is on a
                // different block that may have different angleTypes (or none)
                if ( ( blockAngleType[j] != 0 ) &&                  // this block has angles
                     ( j > 0 || s > 0 ) &&                          // not the first monomer
                     ( j < numBlocks-1 || s < Nblocks[j]-1 ) ) {    // not the final monomer

                    // std::cout << "j: " << j << " s: " << s << " ind: " << ind << std::endl;
                    int i1 = ind-1;
                    int i2 = ind;
                    int i3 = ind+1;

                    // std::cout << "  ivals: " << i1 << " " << i2 << " " << i3 << std::endl;

                    int n1 = nAngles[i1];
                    int n2 = nAngles[i2];
                    int n3 = nAngles[i3];

                    // std::cout << "  n vals: " << n1 << " " << n2 << " " << n3 << std::endl;

                    int index1 = i1*MAXANGLES*3+3*n1;
                    int index2 = i2*MAXANGLES*3+3*n2;
                    int index3 = i3*MAXANGLES*3+3*n3;

                    // std::cout << "  index: " << index1 << " " << index2 << " " << index3 << std::endl;

                    
                    angleGroup[index1+0] = i1;
                    angleGroup[index1+1] = i2;
                    angleGroup[index1+2] = i3;

                    angleGroup[index2+0] = i1;
                    angleGroup[index2+1] = i2;
                    angleGroup[index2+2] = i3;

                    angleGroup[index3+0] = i1;
                    angleGroup[index3+1] = i2;
                    angleGroup[index3+2] = i3;

                    angleType[i1*MAXANGLES+n1] = blockAngleType[j] - 1;
                    angleType[i2*MAXANGLES+n2] = blockAngleType[j] - 1;
                    angleType[i3*MAXANGLES+n3] = blockAngleType[j] - 1;
                    
                    nAngles[i1]++;
                    nAngles[i2]++;
                    nAngles[i3]++;

                    
                }// blockAngleType[j]



                mID[ind] = molecInd;

                // Increment the particle index
                ind++;
                bbIndex++;
            }// s=0:N[j]

        }// j=0:numBlocks; 

        // Increment molecule index
        molecInd++;
    }// i=0:nmolecs


    std::cout << "nstot is " << nstot << " after molecule creation" << std::endl;
}
